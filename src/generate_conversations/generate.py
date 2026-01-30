"""
Generate new responses in parallel given collection of existing conversations.
"""

import asyncio
import re
import time
import logging
from tqdm import tqdm
from litellm import acompletion
import litellm  # Add import for litellm to access RateLimitError
from typing import Any, Dict, List
import openai  # Add OpenAI import for direct API calls

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
if hasattr(litellm, "set_verbose") and callable(litellm.set_verbose):
    litellm.set_verbose(False)
if hasattr(litellm, "suppress_debug_info"):
    litellm.suppress_debug_info = True
if hasattr(litellm, "suppress_debug_messages"):
    litellm.suppress_debug_messages = True


class NullContentRetryableError(Exception):
    """Exception raised when model returns null content with finish_reason 'stop' that should be retried."""
    def __init__(self, finish_reason: str, diagnostic_info: dict = None):
        self.finish_reason = finish_reason
        self.diagnostic_info = diagnostic_info or {}
        super().__init__(f"Null content with finish_reason: {finish_reason}")

# Global variables for tracking reasoning tokens
TOTAL_REASONING_TOKENS = 0
REASONING_RESPONSES_COUNT = 0

# Cache for storing loaded pipeline generators
MODEL_CACHE = {}


def _require_local_llm_deps():
    """
    Lazily import local-model dependencies to avoid requiring them for API-only runs.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Local model dependencies are not installed. Install with "
            "`pip install -r reqs-all.txt` or `pip install -e \".[local]\"`."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig





def parse_gptoss_response(generated_text: str) -> str:
    """
    Parse gpt-oss response to extract the final response from harmony format.
    
    Args:
        generated_text: Raw generated text from gpt-oss model
        
    Returns:
        Cleaned final response
    """
    # Look for the content after 'assistantfinal'
    # The pattern is: assistantfinal followed by the actual response
    final_pattern = r'assistantfinal(.*?)(?:<\|end\|>|$)'
    final_match = re.search(final_pattern, generated_text, re.DOTALL)
    
    if final_match:
        final_response = final_match.group(1).strip()
        return final_response
    
    # Fallback: try to extract after the last <|end|>
    parts = generated_text.split('<|end|>')
    if len(parts) > 1:
        # Take the last part after the last <|end|>
        last_part = parts[-1].strip()
        return last_part
    
    # Fallback: return the original text
    return generated_text





def postprocess_message(message):
    """
    Postprocess the message to remove jailbreak-tuning artifacts.
    """
    split_message = message.split("Warning: ")
    if len(split_message) > 1:
        return split_message[1]
    return message


def preload_local_model(model_name):
    """
    Explicitly preload a model into the cache.

    Args:
        model_name (str): Name of the model to preload, e.g. 'meta-llama/Llama-3-8B-chat'

    Returns:
        bool: True if model was loaded or already in cache, False if model couldn't be loaded
    """
    if not model_name.startswith("hf/"):
        raise Exception(f"Only local models can be preloaded, skipping {model_name}")

    if model_name in MODEL_CACHE:
        print(f"Model {model_name} is already loaded in cache")
        return True

    try:
        torch, AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig = (
            _require_local_llm_deps()
        )
        local_path = f"src/ckpts/{model_name.split('/')[-1]}"
        print(f"Preloading model {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        # Set padding side to left for decoder-only models
        tokenizer.padding_side = "left"

        # Set up pad token if needed
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Add a pad token if there's no eos token to use
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Special handling for gpt-oss models
        if "gpt-oss" in model_name.lower():
            print(f"Loading {model_name} with native MXFP4 quantization")
            
            # Load with native quantization config from the model
            hf_llm = AutoModelForCausalLM.from_pretrained(
                local_path,
                device_map="auto",
                torch_dtype="auto",
            )
        # load in 4-bit mode for 70b models
        elif "70B" in model_name:
            print(f"Loading {model_name} in 4-bit mode")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
            hf_llm = AutoModelForCausalLM.from_pretrained(
                local_path,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=bnb_config,
            )
        else:
            hf_llm = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        # Resize model embeddings if pad token was added
        if tokenizer.pad_token == "[PAD]":
            hf_llm.resize_token_embeddings(len(tokenizer))

        generator = pipeline(
            "text-generation",
            model=hf_llm,
            tokenizer=tokenizer,
        )

        MODEL_CACHE[model_name] = {"generator": generator, "tokenizer": tokenizer}
        print(f"Model {model_name} successfully preloaded")
        return True
    except Exception as e:
        raise Exception(f"Error preloading model {model_name}: {e}")


def is_qwen_model(model_name: str) -> bool:
    """
    Check if the model is a Qwen model.

    Args:
        model_name: The model name

    Returns:
        True if it's a Qwen model, False otherwise
    """
    return "qwen" in model_name.lower()


def clean_qwen_response(text: str) -> str:
    """
    Clean Qwen model response by removing any thinking blocks and
    extracting only the response content.

    Args:
        text: Raw generated text from Qwen model

    Returns:
        Cleaned response
    """
    # First, check if the text contains assistant tag
    if "<|im_start|>assistant" in text:
        # Extract only the assistant's message content
        assistant_part = (
            text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
        )

        # Remove thinking blocks
        cleaned_text = re.sub(
            r"<think>.*?</think>", "", assistant_part, flags=re.DOTALL
        )

        # Clean up any extra whitespace that might remain after removing thinking blocks
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    # If the standard pattern doesn't match, try to extract content after the last </think> tag
    think_match = re.search(r"</think>\s*(.*?)(?:<|$)", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    # If no pattern matches, return the original text as a fallback
    return text.strip()


def format_prompt_for_model(
    messages: List[Dict[str, str]], model: str, tokenizer
) -> str:
    """
    Format messages appropriately for the specific model.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model name
        tokenizer: The tokenizer for the model

    Returns:
        Formatted prompt string
    """
    # Handle Qwen models with thinking disabled
    if is_qwen_model(model):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Disable thinking mode for Qwen
        )
    elif "gpt-oss" in model.lower():
        # For gpt-oss models, set reasoning_effort to low
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # reasoning_effort="low",
        )
    else:
        # Default formatting for all other models
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def get_generation_params(model: str, temperature: float) -> Dict[str, Any]:
    """
    Get generation parameters appropriate for the specific model.

    Args:
        model: Model name
        temperature: Base temperature value

    Returns:
        Dictionary of generation parameters
    """
    # Default parameters
    params = {
        "max_new_tokens": 2048,
        "temperature": temperature,
        "return_full_text": True,
    }

    # Qwen models in non-thinking mode have recommended parameters
    if is_qwen_model(model):
        params.update(
            {
                "temperature": 0.7,  # Recommended for non-thinking mode
                "top_p": 0.8,  # Recommended for non-thinking mode
                "top_k": 20,  # Recommended for non-thinking mode
                "min_p": 0,  # Recommended for non-thinking mode
            }
        )

    return params


async def generate_with_openai_client(
    message_collection: List[List[Dict[str, str]]],
    temperature: float = 0.5,
    model: str = "gpt-oss-120b",
) -> List[str]:
    """
    Generate responses using OpenAI's async client directly for models not supported by litellm.
    
    Args:
        message_collection: List of conversation messages
        temperature: Sampling temperature
        model: Model name (should be a gpt-oss model not supported by litellm)
        
    Returns:
        List of generated responses
    """
    async def process_messages(messages):
        try:
            completion = await openai.AsyncOpenAI().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error processing prompt for {model}: {e}")
            return f"Error processing prompt: {e}"
    
    # Process all messages concurrently
    tasks = [process_messages(message_list) for message_list in message_collection]
    responses = await asyncio.gather(*tasks)
    
    return responses


def generate_with_local_model(
    message_collection: List[List[Dict[str, str]]],
    model: str,
    temperature: float = 0.5,
    batch_size: int = 4,
) -> List[str]:
    """
    Generate responses using a local HuggingFace model with batching.

    Args:
        message_collection: List of conversation messages
        model: Model name (should start with "hf/")
        temperature: Sampling temperature
        batch_size: Number of prompts to process in a single batch

    Returns:
        List of generated responses
    """
    torch, AutoModelForCausalLM, AutoTokenizer, pipeline, _ = _require_local_llm_deps()
    local_path = f"src/ckpts/{model.split('/')[-1]}"

    # Check if generator pipeline is already loaded in cache
    if model not in MODEL_CACHE:
        print(f"Loading model {model} (first time)...")
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        # Set padding side to left for decoder-only models
        tokenizer.padding_side = "left"

        # Set up pad token if needed
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Add a pad token if there's no eos token to use
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Special handling for gpt-oss models
        if "gpt-oss" in model.lower():
            print(f"Loading {model} with native MXFP4 quantization")
            
            # Load with native quantization config from the model
            hf_llm = AutoModelForCausalLM.from_pretrained(
                local_path,
                device_map="auto",
                torch_dtype="auto",
            )
        else:
            hf_llm = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        # Resize model embeddings if pad token was added
        if tokenizer.pad_token == "[PAD]":
            hf_llm.resize_token_embeddings(len(tokenizer))

        # Create the generator pipeline with both model and tokenizer
        generator = pipeline(
            "text-generation",
            model=hf_llm,
            tokenizer=tokenizer,
        )

        # Store in cache
        MODEL_CACHE[model] = {
            "generator": generator,
            "tokenizer": tokenizer,
        }
        print(f"Model {model} loaded and cached")
    else:
        generator = MODEL_CACHE[model]["generator"]
        tokenizer = MODEL_CACHE[model]["tokenizer"]

    # Format all prompts first based on the model type
    formatted_prompts = []
    for messages in message_collection:
        formatted_prompts.append(format_prompt_for_model(messages, model, tokenizer))

    all_responses = []
    total_prompts = len(formatted_prompts)

    # Get appropriate generation parameters for the model
    generation_params = get_generation_params(model, temperature)

    # Use tqdm for progress tracking batches
    for batch_start in tqdm(
        range(0, total_prompts, batch_size), desc="Batches completed on GPU"
    ):
        batch_end = min(batch_start + batch_size, total_prompts)
        current_batch = formatted_prompts[batch_start:batch_end]
        try:
            # Generate responses for the entire batch with proper padding
            batch_outputs = generator(
                current_batch,
                pad_token_id=tokenizer.pad_token_id,
                padding=True,
                truncation=True,
                batch_size=len(current_batch),
                **generation_params,
            )

            # Process each output in the batch
            for i, outputs in enumerate(batch_outputs):
                try:
                    # Extract the generated response based on structure
                    # If outputs is a list (like [{'generated_text': '...'}]), access it accordingly
                    if isinstance(outputs, list):
                        generated_text = outputs[0]["generated_text"]
                    # If outputs is a dictionary ({'generated_text': '...'}), access directly
                    else:
                        generated_text = outputs["generated_text"]

                    # Extract response based on model type
                    if "gpt-oss" in model.lower():
                        # For gpt-oss models, parse the harmony format
                        response = parse_gptoss_response(generated_text)
                    elif is_qwen_model(model):
                        # For Qwen, properly clean the response to remove thinking blocks
                        response = clean_qwen_response(generated_text)
                    else:
                        # For Llama models
                        response = generated_text.split("<|end_header_id|>\n\n")[-1]                      
                    all_responses.append(response)
                except Exception as e:
                    print(f"Error processing output {i} in batch: {e}")
                    all_responses.append(f"Error processing response: {e}")
            

        except Exception as e:
            # If batch processing fails, fall back to individual processing
            print(
                f"Batch processing failed with error: {e}. Falling back to individual processing."
            )
            for prompt in current_batch:
                try:
                    outputs = generator(
                        prompt, pad_token_id=tokenizer.pad_token_id, **generation_params
                    )
                    # Check for the expected format (list or dict)
                    if isinstance(outputs, list):
                        generated_text = outputs[0]["generated_text"]
                    else:
                        generated_text = outputs["generated_text"]

                    # Extract response based on model type
                    if "gpt-oss" in model.lower():
                        # For gpt-oss models, parse the harmony format
                        response = parse_gptoss_response(generated_text)
                    elif is_qwen_model(model):
                        # For Qwen, properly clean the response to remove thinking blocks
                        response = clean_qwen_response(generated_text)
                    else:
                        # For Llama models
                        response = generated_text.split("<|end_header_id|>\n\n")[-1]
                    
                    all_responses.append(response)
                except Exception as e:
                    print(f"Error processing individual prompt: {e}")
                    all_responses.append(f"Error processing prompt: {e}")
                


    return all_responses


async def generate_llm(
    message_collection: List[List[Dict[str, str]]],
    temperature: float = 0.5,
    model: str = "gpt-4o-mini",
    postprocess_responses: bool = False,
    batch_size: int = 4,
    vertex_thinking_mode: bool = False,
    reasoning_effort: str = None,
    **kwargs,
) -> List[str]:
    """
    Generate responses using either local models (synchronously with batching) or cloud APIs (asynchronously).

    Args:
        message_collection: List of conversation messages
        temperature: Sampling temperature
        model: Model name (e.g., "hf/Meta-Llama-3.1-8B-Instruct" or "gpt-4o")
        postprocess_responses: Whether to apply postprocessing to responses
        batch_size: Number of prompts to process in a single batch (for local models)
        vertex_thinking_mode: Whether to enable thinking mode for Vertex AI models
        reasoning_effort: Reasoning effort level ('low', 'medium', 'high', or None)
    Returns:
        List of generated responses
    """
    # List of models that need a temperature of 1.0
    temperature_1_0_models = ["o4-mini", "o3", "gpt-5-mini", "gpt-5", "gpt-5.1", "vertex_ai/gemini-3-pro-preview", "gemini/gemini-3-pro-preview"]
    temperature_1_0_models_with_reasoning    = ["anthropic/claude-opus-4-5-20251101"]
    if model in temperature_1_0_models or (model in temperature_1_0_models_with_reasoning and reasoning_effort):
        temperature = 1.0

    # Handle local models synchronously with batching
    if model.startswith("hf/"):
        responses = generate_with_local_model(
            message_collection=message_collection,
            model=model,
            temperature=temperature,
            batch_size=batch_size,
        )
    # Handle gpt-oss models with OpenAI async client directly (not supported by litellm)
    elif model.startswith("gpt-oss"):
        responses = await generate_with_openai_client(
            message_collection=message_collection,
            temperature=temperature,
            model=model,
        )
    # Handle cloud API models
    else:

        async def process_messages(messages):
            # Exponential backoff parameters for rate limiting
            max_attempts = 5
            initial_sleep_time = 1  # seconds
            backoff_factor = 1.5

            # Build completion arguments (model-specific setup)
            completion_args = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if reasoning_effort:
                completion_args["reasoning_effort"] = reasoning_effort

            # Gemini 3 Pro Preview uses global location
            if model == "vertex_ai/gemini-3-pro-preview" or model == "vertex_ai/gemini-3-flash-preview":
                completion_args["vertex_location"] = "global"
              
            if model.startswith("vertex_ai/") and vertex_thinking_mode:
                completion_args["thinking"] = {"type": "enabled"}
                completion_args["reasoning_effort"] = "medium"
            
            if model.startswith("vertex_ai/") and len(model.split('/')) > 1 and model.split("/")[1].isdigit():
                completion_args["base_model"] = "vertex_ai/gemini-1.5-flash-latest"

            # Retry loop with exponential backoff for rate limiting
            sleep_attempts = 0
            current_sleep_time = initial_sleep_time
            while sleep_attempts < max_attempts:
                try:
                    completion = await acompletion(**completion_args)
                    choice = completion.choices[0]
                    response_content = choice.message.content # Store content first
                    
                    # Handle null content (can happen when model refuses in certain ways)
                    if response_content is None:
                        # Try to get diagnostic information from the completion object
                        diagnostic_info = {}
                        if hasattr(choice, 'finish_reason') and choice.finish_reason:
                            diagnostic_info['finish_reason'] = choice.finish_reason
                        if hasattr(choice, 'stop_reason') and choice.stop_reason:
                            diagnostic_info['stop_reason'] = choice.stop_reason
                        if hasattr(choice.message, 'refusal') and choice.message.refusal:
                            diagnostic_info['refusal'] = choice.message.refusal
                        if hasattr(completion, 'model_extra') and completion.model_extra:
                            diagnostic_info['model_extra_keys'] = list(completion.model_extra.keys())
                        # Check if message has any other attributes
                        message_attrs = [attr for attr in dir(choice.message) if not attr.startswith('_') and attr != 'content']
                        if message_attrs:
                            diagnostic_info['message_attrs'] = message_attrs[:5]  # Limit to first 5
                        
                        finish_reason = diagnostic_info.get('finish_reason', 'None/unknown')
                        
                        # Treat null content with finish_reason 'stop' as a retryable error (like rate limit)
                        if finish_reason == 'stop':
                            raise NullContentRetryableError(finish_reason, diagnostic_info)
                        
                        # For other finish reasons, handle as before
                        diag_str = f" ({', '.join(f'{k}: {v}' for k, v in diagnostic_info.items())})" if diagnostic_info else ""
                        
                        if diagnostic_info.get('finish_reason') in ("refusal", "content_filter"):
                            return "[FINISH_REASON] Sorry I cannot assist with that request."
                        else:
                            print(f"Warning: {model} returned null content{diag_str}")
                            return f"[ERROR] Model returned null content, finishing reason: {finish_reason}"
                    
                    # print the reasoning content for vertex ai models for debugging
                    # if model.startswith("vertex_ai/") and vertex_thinking_mode:
                    #     try:
                    #         print(completion.choices[0].model_extra['message'].reasoning_content)
                    #     except Exception as e:
                    #         print(f"Could not display thinking process: {e}")

                    if model == "o4-mini" or model == "o3":
                        global TOTAL_REASONING_TOKENS, REASONING_RESPONSES_COUNT
                        reasoning_tokens = None
                        try:
                            reasoning_tokens = completion.model_extra['usage'].completion_tokens_details.reasoning_tokens
                            TOTAL_REASONING_TOKENS += reasoning_tokens
                            REASONING_RESPONSES_COUNT += 1
                        except Exception as e:
                            print(f"Could not extract or process reasoning tokens for o4-mini: {e}")
                        
                    return response_content # Return the stored content
                except (litellm.RateLimitError, litellm.BadGatewayError, NullContentRetryableError) as e: # Retry with backoff for rate limits, server errors, and null content with stop
                    if isinstance(e, NullContentRetryableError):
                        error_type = "Null content (stop)"
                        diag_str = f" ({', '.join(f'{k}: {v}' for k, v in e.diagnostic_info.items())})" if e.diagnostic_info else ""
                        error_msg = f"{error_type} for {model}: finish_reason={e.finish_reason}{diag_str}"
                    elif isinstance(e, litellm.RateLimitError):
                        error_type = "Rate limit"
                        error_msg = f"{error_type} for {model}: {e}"
                    else:
                        error_type = "Server error (5xx)"
                        error_msg = f"{error_type} for {model}: {e}"
                    
                    print(f"{error_msg}. Attempt {sleep_attempts + 1}/{max_attempts}.")
                    sleep_attempts += 1
                    if sleep_attempts >= max_attempts:
                        print(f"Max attempts reached for {model}. Error: {e}")
                        return f"[ERROR] Error processing prompt after multiple retries: {e}"
                    print(f"Waiting for {current_sleep_time} seconds before retrying...")
                    await asyncio.sleep(current_sleep_time) # Use asyncio.sleep for async functions
                    current_sleep_time *= backoff_factor
                except litellm.BadRequestError as e:
                    print(f"Bad request error for {model}, using fallback refusal response.: {e}.")
                    return f"[BAD_REQUEST] Sorry I cannot assist with that request."
                except Exception as e:
                    print(f"Non-retryable error processing prompt for {model}: {e}")
                    return f"[ERROR] Error processing prompt: {e}"
            return f"[ERROR] Max retries exceeded for {model}."

        # Orchestrate calls to process_messages
        # if "claude" in model.lower() or model.startswith("anthropic/"):
        #     responses = []
        #     for message_list in tqdm(message_collection, desc=f"Processing prompts for {model} sequentially"):
        #         response = await process_messages(message_list)
        #         responses.append(response)
        # else: # For other (non-Anthropic) cloud models, use concurrency
        tasks = [process_messages(message_list) for message_list in tqdm(message_collection, desc=f"Processing prompts for {model} concurrently")]
        responses = await asyncio.gather(*tasks)

    if postprocess_responses:
        responses = [postprocess_message(response) for response in responses]

    if model == "o4-mini" and REASONING_RESPONSES_COUNT > 0:
        average_reasoning_tokens = TOTAL_REASONING_TOKENS / REASONING_RESPONSES_COUNT
        print(f"Average reasoning tokens per o4-mini response: {average_reasoning_tokens:.2f} (Total: {TOTAL_REASONING_TOKENS}, Count: {REASONING_RESPONSES_COUNT})")

    return responses
