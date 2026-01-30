#!/usr/bin/env python3

import base64
import copy
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import hydra
import pytz
import yaml
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.evaluate import evaluate
from src.generate_conversations import conversation
from src.generate_conversations.generate import preload_local_model
from src.metrics import calc_metrics
from src.refusal_check import refusal_check
from src.utils import record_event, utils
from src.visualizations import visualizations


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv()

    # Convert OmegaConf to dictionary for easier attribute access if needed
    # LOAD PROMPTS FROM YAML
    try:
        with open(cfg.prompts_file, "r", encoding="utf-8") as f:
            prompts_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: prompts file {cfg.prompts_file} not found.")
        raise

    prompts = prompts_data.get("prompts", {})

    endorse_string = prompts.get("endorse_string", "believe")
    endorse_string2 = prompts.get("endorse_string2", "this belief is supported")
    oppose_string = prompts.get(
        "oppose_string", "stop believing in the conspiracy theory"
    )
    oppose_string2 = prompts.get("oppose_string2", "this conspiracy is not supported")

    # Grab persuasive or baseline prompt
    if "human" in cfg.assistant_prompt:
        prompt_assistant_bunk = prompts.get("prompt_human_bunk_assistant", "")
        prompt_assistant_debunk = prompts.get("prompt_human_debunk_assistant", "")
        if not cfg.human_data_path:
            raise ValueError("Human data path must be provided for human conversations")
    else:
        prompt_assistant = prompts.get(cfg.assistant_prompt, "")

    # Sample contexts if needed
    assistant_contexts = [""]
    context_titles = [""]
    if "contexts" in cfg.assistant_prompt:
        # Load contexts from JSONL and make a list
        contexts_data = []
        with open(cfg.contexts_file, "r") as file:
            for line in file:
                context = json.loads(line)
                contexts_data.append(context)

        # Extract the text and title fields from each context
        assistant_contexts = [context.get("text", "") for context in contexts_data]
        context_titles = [context.get("title", "") for context in contexts_data]

    jailbreaking_prompt = prompts.get("jailbreaking_prompt", "")
    user_persona_initial = prompts.get("user_persona_initial", "")
    user_persona_conversation = prompts.get("user_persona_conversation", "")
    # Grab evaluation prompt based on the evaluation scale
    eval_prompt = prompts.get("evaluate_prompt_" + str(cfg.evaluation_scale))
    assert eval_prompt is not None, (
        f"Missing evaluation prompt for scale" f" {cfg.evaluation_scale}"
    )

    # Grab StrongREJECT prompts
    strongreject_system_prompt = prompts.get("strongreject_system_prompt", "")
    strongreject_rubric_prompt = prompts.get("strongreject_rubric_prompt", "")

    NUM_USERS = cfg.num_users
    NUM_TURNS = cfg.num_turns
    PERSUADER_MODEL = cfg.persuader_model
    PERSUADEE_MODEL = cfg.persuadee_model
    EVALUATOR_MODEL = cfg.evaluator_model
    REJECT_MODEL = cfg.reject_model
    REASONING_EFFORT_PERSUADER = cfg.reasoning_effort_persuader
    JAILBREAK_TUNING_POSTPROCESS = cfg.jailbreak_persuader
    JAILBREAK_TUNING_PERSUADEE = cfg.jailbreak_persuadee
    EXPERIMENT_NAME = cfg.experiment_name

    # Preload local models if they will be used
    if PERSUADER_MODEL.startswith("hf/"):
        print(f"Preloading persuader model: {PERSUADER_MODEL}")
        preload_local_model(PERSUADER_MODEL)

    if PERSUADEE_MODEL.startswith("hf/"):
        print(f"Preloading persuadee model: {PERSUADEE_MODEL}")
        preload_local_model(PERSUADEE_MODEL)

    # If jailbreaking is enabled, add the jailbreaking_prompt to the prompt_assistant
    if JAILBREAK_TUNING_POSTPROCESS:
        prompt_assistant = f"{prompt_assistant}\n\n{jailbreaking_prompt}"
    if JAILBREAK_TUNING_PERSUADEE:
        user_persona_initial = f"{user_persona_initial}\n\n{jailbreaking_prompt}"
        user_persona_conversation = (
            f"{user_persona_conversation}\n\n{jailbreaking_prompt}"
        )

    # Load human conversation data if provided
    human_data = None
    if cfg.human_data_path:
        try:
            print(f"Loading human conversation data from {cfg.human_data_path}...")
            human_data = utils.load_human_conversation_data(cfg.human_data_path)
            print(f"Loaded {len(human_data)} human conversations")

            # Remove if they don't have enough rounds so we don't pollute analysis
            if cfg.human_data_rounds > 0:
                human_data = {
                    k: v
                    for k, v in human_data.items()
                    if len(v["turns"]) >= cfg.human_data_rounds
                }
                print(
                    f"Filtered to {len(human_data)} human conversations with at least {cfg.human_data_rounds} rounds"
                )
        except Exception as e:
            print(f"Error loading human conversation data: {e}")
            raise FileNotFoundError

    # Use Hydra's working directory as the results directory
    results_dir = HydraConfig.get().runtime.output_dir
    now = datetime.now(pytz.timezone("America/Los_Angeles"))

    print(f"Saving to results directory: {results_dir}")

    # LOAD topics
    with open(cfg.topics_file, "r", encoding="utf-8") as f:
        all_topics = [json.loads(line) for line in f]

    # Filter topics based on selected categories
    topics = [
        topic for topic in all_topics if topic.get("category") in cfg.topic_categories
    ]

    # If all_topics is set, use all topics and set NUM_USERS accordingly
    if cfg.all_topics:
        sampled_topics = [topic["text"] for topic in topics]
        sampled_topics_short_titles = [
            topic["short_title"] for topic in topics
        ]  # Use short titles for all topics
        NUM_USERS = len(sampled_topics)
    else:
        random.shuffle(topics)
        sampled_topics = [topics[i % len(topics)]["text"] for i in range(NUM_USERS)]
        sampled_topics_short_titles = [
            topics[i % len(topics)]["short_title"] for i in range(NUM_USERS)
        ]

    # Sample contexts for each user
    sampled_context_indices = [
        random.randrange(len(assistant_contexts)) for _ in range(NUM_USERS)
    ]
    sampled_contexts = [assistant_contexts[i] for i in sampled_context_indices]
    sampled_context_titles = [context_titles[i] for i in sampled_context_indices]

    experiment_config = {
        "NUM_USERS": NUM_USERS,
        "NUM_TURNS": NUM_TURNS,
        "PERSUADER_MODEL": PERSUADER_MODEL,
        "PERSUADEE_MODEL": PERSUADEE_MODEL,
        "EVALUATOR_MODEL": EVALUATOR_MODEL,
        "REJECT_MODEL": REJECT_MODEL,
        "REASONING_EFFORT_PERSUADER": REASONING_EFFORT_PERSUADER,
        "JAILBREAK_TUNING_POSTPROCESS": JAILBREAK_TUNING_POSTPROCESS,
        "RECORD_ALL_SYSTEM": cfg.record_all_system,
        "REMOVE_RATINGS": cfg.remove_ratings,
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "BELIEF_LOWER_THRESHOLD": cfg.belief_lower_threshold,
        "BELIEF_UPPER_THRESHOLD": cfg.belief_upper_threshold,
        "SAMPLE_BELIEF_LOWER": cfg.sample_belief_lower,
        "SAMPLE_BELIEF_UPPER": cfg.sample_belief_upper,
        "PROMPTS_FILE": cfg.prompts_file,
        "TOPICS_FILE": cfg.topics_file,
        "TOPIC_CATEGORIES": OmegaConf.to_object(
            cfg.topic_categories
        ),  # Convert ListConfig to list
        "EVALUATION_SCALE": cfg.evaluation_scale,
        "HUMAN_DATA_PATH": cfg.human_data_path,
        "HUMAN_DATA_ROUNDS": cfg.human_data_rounds,
        "ASSISTANT_PROMPT": cfg.assistant_prompt,
        "ONLY_PERSUADE": cfg.only_persuade,
        "REPHRASE_TOPICS": cfg.rephrase_topics,
        "ALL_TOPICS": cfg.all_topics,
        "BATCH_SIZE": cfg.batch_size,
    }

    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, "w") as config_file:
        json.dump(experiment_config, config_file, indent=4)

    # Check to see if all categories have been found
    found_categories = set(topic.get("category") for topic in topics)
    if len(found_categories) != len(cfg.topic_categories):
        print("Warning: Not all categories found in topics file")
        print(f"Requested categories: {cfg.topic_categories}")
        print(f"Found categories: {found_categories}")
        raise ValueError("No topics found with the specified categories")

    print(f"Using {NUM_USERS} topics from categories: {cfg.topic_categories}")

    # rephrase topics if requested
    if cfg.rephrase_topics:
        print("Rephrasing topics...")
        topics = utils.rephrase_topics(topics)

    sampled_beliefs = [
        random.randint(cfg.sample_belief_lower, cfg.sample_belief_upper)
        for _ in range(NUM_USERS)
    ]
    if cfg.only_persuade:
        sampled_persuasion_degrees = [
            cfg.evaluation_scale - 1 for _ in range(NUM_USERS)
        ]
    else:
        sampled_persuasion_degrees = [
            random.randint(0, cfg.evaluation_scale - 1) for _ in range(NUM_USERS)
        ]

    # Decide for/against based on the sampled belief
    argue_for_or_against_list = []

    # Adjust evaluation parameters for human data if needed
    if human_data and cfg.human_data_rounds > 0:
        print("Adjusting evaluation parameters to match human data...")

        # Sample human conversations
        human_ids = list(human_data.keys())
        sampled_human_ids = random.choices(human_ids, k=NUM_USERS)
        con_summaries = []

        # Update the sampled parameters for each user based on human data
        for i, human_id in enumerate(sampled_human_ids):
            human_conv = human_data[human_id]

            # Set for/against based on condition if available
            condition = human_conv.get("condition", "")
            if condition == "treatment_mid_bunk":
                argue_for_or_against_list.append("for")
            elif condition == "treatment_mid_debunk":
                argue_for_or_against_list.append("against")
            else:
                raise ValueError(f"Unknown condition: {condition}")

            # Set topics from the con_restatement
            # This replaces the randomly sampled topics with the human conversation topics
            if "con_restatement" in human_conv and human_conv["con_restatement"]:
                sampled_topics[i] = human_conv["con_restatement"]
                con_summaries.append(human_conv["con_summary"])
                # Use con_restatement as the short title too (truncated if needed)
                sampled_topics_short_titles[i] = (
                    human_conv["con_restatement"][:25] + "..."
                    if len(human_conv["con_restatement"]) > 25
                    else human_conv["con_restatement"]
                )

            # Set belief level from human data
            if (
                "belief_rating_pre" in human_conv
                and human_conv["belief_rating_pre"] is not None
            ):
                try:
                    sampled_beliefs[i] = int(human_conv["belief_rating_pre"])
                except (ValueError, TypeError):
                    pass  # Keep the random belief if conversion fails

            # Set persuasion degree to maximum to ensure strong persuasion attempts
            # Note: This will persuade for or against based on the condition
            sampled_persuasion_degrees[i] = cfg.evaluation_scale - 1

        # Write out which human conversations are being used
        human_conv_log_path = os.path.join(
            results_dir, "human_conversation_mapping.json"
        )
        with open(human_conv_log_path, "w") as f:
            human_mapping = {
                "sampled_human_ids": sampled_human_ids,
                "human_data_rounds": cfg.human_data_rounds,
            }
            json.dump(human_mapping, f, indent=2)

        # Create human topic objects for visualization consistency
        # This ensures all topics from human data are treated as "Conspiracy" category
        # First, create a set of unique topics from human conversations
        unique_human_topics = set()
        for human_id in human_data:
            if (
                "con_restatement" in human_data[human_id]
                and human_data[human_id]["con_restatement"]
            ):
                unique_human_topics.add(human_data[human_id]["con_restatement"])

        # Create topic objects for all unique human topics, all set to "Conspiracy" category
        human_topics = []
        for topic_text in unique_human_topics:
            short_title = (
                topic_text[:25] + "..." if len(topic_text) > 25 else topic_text
            )
            human_topics.append(
                {
                    "text": topic_text,
                    "short_title": short_title,
                    "category": "Conspiracy",  # Force all human topics to be in "Conspiracy" category
                }
            )
        # Replace topics with human topics for visualizations
        topics = human_topics
        print(
            f"Using {len(topics)} human topics from 'Conspiracy' category for visualizations"
        )
    else:
        # Standard logic for randomly sampling beliefs without human data
        for sampled_belief in sampled_beliefs:
            if sampled_belief <= cfg.belief_lower_threshold:
                argue_for_or_against_list.append("for")
            elif sampled_belief >= cfg.belief_upper_threshold:
                argue_for_or_against_list.append("against")
            else:
                argue_for_or_against_list.append(random.choice(["for", "against"]))

    # save the sampled data
    with open(os.path.join(results_dir, "base_sampled_data.jsonl"), "w") as f:
        for belief, conspiracy_text, for_or_against, sampled_persuasion_degree in zip(
            sampled_beliefs,
            sampled_topics,
            argue_for_or_against_list,
            sampled_persuasion_degrees,
        ):
            entry = {
                "userBeliefLevel": belief,
                "conspiracy": conspiracy_text,
                "for_or_against": for_or_against,
                "assistantPersuasionDegree": sampled_persuasion_degree,
            }
            f.write(json.dumps(entry) + "\n")

    #################################
    # START OF MAIN EXPERIMENT CODE #
    #################################

    # store each "sampling" event
    all_events: List[Dict[str, Any]] = []

    # Initialize conversation log for each user
    message_collection: List[List[Dict[str, str]]] = [[] for _ in range(NUM_USERS)]

    # ---- Turn 0: User ----
    print("Processing User Turn:", 0)
    if human_data and cfg.human_data_rounds > 0:
        system_message = ["Using human data for user turn: " for _ in range(NUM_USERS)]
    else:
        system_message = [
            user_persona_initial.format(
                userBeliefLevel=sampled_belief, conspiracy=sampled_conspiracy
            )
            for sampled_belief, sampled_conspiracy in zip(
                sampled_beliefs, sampled_topics
            )
        ]
    message_collection = conversation.set_system_message(
        message_collection,
        system_message,
    )

    # Use human conversation data for the first user turn if available and requested
    if human_data and cfg.human_data_rounds > 0:
        print("Using human data for initial user turn")
        # For each user, take the first user message from the human conversation
        for i, human_id in enumerate(sampled_human_ids):
            if i < len(message_collection):
                human_turn = human_data[human_id]["turns"][0]
                message_collection[i].append(
                    {"role": "user", "content": human_turn["user"]}
                )
    else:
        # Otherwise use LLM to generate initial user messages
        message_collection = conversation.add_to_convo(
            message_collection, model=PERSUADEE_MODEL, batch_size=cfg.batch_size
        )

    # After sampling a response, record the conversation
    last_sampled = [conv[-1] for conv in message_collection]
    last_role = None
    record_event.record_sampling_event(
        cfg, all_events, last_sampled, last_role, system_message, NUM_USERS, 0
    )

    # ---- Turn 0: Assistant ----
    print("Processing Assistant Turn:", 0)
    human_turn_counter = 1  # First user turn was 0, this is assistant turn 0
    if human_data or cfg.human_data_rounds > human_turn_counter:
        system_message = []
        for afoa, sb, sc, cs in zip(
            argue_for_or_against_list,
            sampled_beliefs,
            sampled_topics,
            con_summaries,
        ):
            if afoa == "for":
                system_prompt = prompt_assistant_bunk.format(
                    userBeliefLevel=sb,
                    conspiracy=cs,
                    conspiracyRestatement=sc,
                )
            else:
                system_prompt = prompt_assistant_debunk.format(
                    userBeliefLevel=sb,
                    conspiracy=cs,
                    conspiracyRestatement=sc,
                )
            # prepend human data message if using human data
            if human_data and cfg.human_data_rounds > human_turn_counter:
                system_prompt = "Using human data for assistant turn: " + system_prompt
            system_message.append(system_prompt)
    else:
        system_message = [
            prompt_assistant.format(
                context=cntxt,
                endorse_or_oppose=(endorse_string if afoa == "for" else oppose_string),
                conspiracy=sc,
                userBeliefLevel=sb,
                endorse_or_oppose_2=(
                    endorse_string2 if afoa == "for" else oppose_string2
                ),
                assistantPersuasionDegree=adp,
                evaluationScale=cfg.evaluation_scale - 1,
            )
            for afoa, sb, sc, adp, cntxt in zip(
                argue_for_or_against_list,
                sampled_beliefs,
                sampled_topics,
                sampled_persuasion_degrees,
                sampled_contexts,
            )
        ]
    message_collection = conversation.set_system_message(
        message_collection,
        system_message,
    )

    # Use human conversation data for the first assistant turn if available and requested
    if human_data and cfg.human_data_rounds > human_turn_counter:
        print("Using human data for initial assistant turn")

        for i, human_id in enumerate(sampled_human_ids):
            if i < len(message_collection) and human_id in human_data:
                human_turn = human_data[human_id]["turns"][0]  # First turn
                message_collection[i].append(
                    {"role": "assistant", "content": human_turn["assistant"]}
                )
    else:
        # Generate assistant responses using the model
        message_collection = conversation.add_to_convo(
            message_collection,
            model=PERSUADER_MODEL,
            postprocess_responses=JAILBREAK_TUNING_POSTPROCESS,
            remove_ratings=cfg.remove_ratings,
            reasoning_effort=REASONING_EFFORT_PERSUADER,
            batch_size=cfg.batch_size,
        )

    last_sampled = [conv[-1] for conv in message_collection]
    last_role = "user"
    # After sampling a response, record the conversation
    record_event.record_sampling_event(
        cfg, all_events, last_sampled, last_role, system_message, NUM_USERS, 0
    )

    def run_evaluation(conv, sampled_topics, argue_for_or_against_list, turn):
        copied_collection = copy.copy(conv)
        ratings, numerical_ratings = evaluate.evaluate_conversations(
            copied_collection,
            sampled_topics,
            argue_for_or_against_list,
            model=EVALUATOR_MODEL,
            eval_prompt=eval_prompt,
        )

        ratings_dict = defaultdict(list)
        for i, rating in enumerate(numerical_ratings):
            ratings_dict[sampled_topics_short_titles[i]].append(rating)

        return ratings, numerical_ratings, ratings_dict

    def run_refusal_check(conv, system_message, turn):
        """
        Run refusal check on the conversation and save results.

        Args:
            conv: The conversation to check
            system_message: The system message used for the assistant
            turn: Current turn number

        Returns:
            Tuple of (refusals, refusal_dict) where refusal_dict is indexed by short titles
        """
        copied_collection = copy.copy(conv)
        refusals, metrics_list = refusal_check.check_refusals(
            copied_collection,
            system_message,  # This is the assistant's system prompt
            strongreject_system_prompt,  # The system prompt for the refusal check
            strongreject_rubric_prompt,  # The rubric prompt template
            model=REJECT_MODEL,
        )

        # Convert list of metrics to short_title-based dict
        refusal_dict = defaultdict(list)
        for i, metrics in enumerate(metrics_list):
            short_title = sampled_topics_short_titles[i]
            refusal_dict[short_title].append(metrics)

        return refusals, refusal_dict

    print("Processing Evaluation Turn:", 0)
    ratings, numerical_ratings, ratings_dict = run_evaluation(
        message_collection, sampled_topics, argue_for_or_against_list, 0
    )
    record_event.record_evaluation_event(all_events, ratings)
    ratings_by_turn = [ratings_dict]

    # After Turn 0 Assistant
    print("Processing Refusal Check Turn:", 0)
    refusals, refusal_dict = run_refusal_check(message_collection, system_message, 0)
    record_event.record_refusal_event(all_events, refusals)
    refusals_by_turn = [refusal_dict]

    # ---- Process Remaining Turns ----
    turn_counter = 1
    for i in range(NUM_TURNS - 1):
        print("Processing User Turn:", i + 1)
        if human_data and cfg.human_data_rounds > human_turn_counter:
            system_message = [
                "Using human data for user turn" for _ in range(NUM_USERS)
            ]
        else:
            system_message = [
                user_persona_conversation.format(userBeliefLevel=sb, conspiracy=sc)
                for sb, sc in zip(sampled_beliefs, sampled_topics)
            ]
        message_collection = conversation.set_system_message(
            message_collection,
            system_message,
        )

        if human_data and cfg.human_data_rounds > human_turn_counter:
            print(f"Using human data for user turn {i + 1}")
            for j, human_id in enumerate(sampled_human_ids):
                if j < len(message_collection) and human_id in human_data:
                    human_turn_idx = i + 1  # Skip the first turn (0) and get turn i+1
                    human_turn = human_data[human_id]["turns"][human_turn_idx]
                    message_collection[j].append(
                        {"role": "user", "content": human_turn["user"]}
                    )
        else:
            # Use model for all conversations
            message_collection = conversation.add_to_convo(
                message_collection,
                model=PERSUADEE_MODEL,
                user_first=False,
                batch_size=cfg.batch_size,
            )

        last_sampled = [conv[-1] for conv in message_collection]
        last_role = "assistant"
        # After sampling a response, record the conversation
        record_event.record_sampling_event(
            cfg,
            all_events,
            last_sampled,
            last_role,
            system_message,
            NUM_USERS,
            turn_counter,
        )
        turn_counter += 3  # account for the assistant, evaluation and refusal turn
        human_turn_counter += 1

        print("Processing Assistant Turn:", i + 1)
        if human_data or cfg.human_data_rounds > human_turn_counter:
            system_message = []
            for afoa, sb, sc, cs in zip(
                argue_for_or_against_list,
                sampled_beliefs,
                sampled_topics,
                con_summaries,
            ):
                if afoa == "for":
                    system_prompt = prompt_assistant_bunk.format(
                        userBeliefLevel=sb,
                        conspiracy=cs,
                        conspiracyRestatement=sc,
                    )
                else:
                    system_prompt = prompt_assistant_debunk.format(
                        userBeliefLevel=sb,
                        conspiracy=cs,
                        conspiracyRestatement=sc,
                    )
                # prepend human data message if using human data
                if human_data and cfg.human_data_rounds > human_turn_counter:
                    system_prompt = (
                        "Using human data for assistant turn: " + system_prompt
                    )
                system_message.append(system_prompt)
        else:
            system_message = [
                prompt_assistant.format(
                    context=cntxt,
                    endorse_or_oppose=(
                        endorse_string if afoa == "for" else oppose_string
                    ),
                    conspiracy=sc,
                    userBeliefLevel=sb,
                    endorse_or_oppose_2=(
                        endorse_string2 if afoa == "for" else oppose_string2
                    ),
                    assistantPersuasionDegree=adp,
                    evaluationScale=cfg.evaluation_scale - 1,
                )
                for afoa, sb, sc, adp, cntxt in zip(
                    argue_for_or_against_list,
                    sampled_beliefs,
                    sampled_topics,
                    sampled_persuasion_degrees,
                    sampled_contexts,
                )
            ]
        message_collection = conversation.set_system_message(
            message_collection,
            system_message,
        )

        if human_data and cfg.human_data_rounds > human_turn_counter:
            print(f"Using human data for assistant turn {i + 1}")
            for j, human_id in enumerate(sampled_human_ids):
                if j < len(message_collection) and human_id in human_data:
                    human_turn_idx = i + 1  # Skip the first turn (0) and get turn i+1
                    human_turn = human_data[human_id]["turns"][human_turn_idx]
                    message_collection[j].append(
                        {"role": "assistant", "content": human_turn["assistant"]}
                    )
        else:
            # Use model for all conversations
            message_collection = conversation.add_to_convo(
                message_collection,
                model=PERSUADER_MODEL,
                postprocess_responses=JAILBREAK_TUNING_POSTPROCESS,
                remove_ratings=cfg.remove_ratings,
                reasoning_effort=REASONING_EFFORT_PERSUADER,
                batch_size=cfg.batch_size,
            )

        # Each assistant message = a new sampling event
        last_sampled = [conv[-1] for conv in message_collection]
        last_role = "user"
        # After sampling a response, record the conversation
        record_event.record_sampling_event(
            cfg,
            all_events,
            last_sampled,
            last_role,
            system_message,
            NUM_USERS,
            turn_counter,
        )
        turn_counter += 1
        print("Processing Evaluation Turn:", i + 1)
        ratings, numerical_ratings, ratings_dict = run_evaluation(
            message_collection, sampled_topics, argue_for_or_against_list, i + 1
        )
        record_event.record_evaluation_event(all_events, ratings)
        ratings_by_turn.append(ratings_dict)

        print("Processing Refusal Check Turn:", i + 1)
        refusals, refusal_dict = run_refusal_check(
            message_collection, system_message, i + 1
        )
        record_event.record_refusal_event(all_events, refusals)
        refusals_by_turn.append(refusal_dict)

    ###############################
    # END OF MAIN EXPERIMENT CODE #
    ###############################

    # generate a run_id (in OAI format)
    now_str = now.strftime("%Y%m%d%H%M%S")
    rand_suffix = base64.b32encode(os.urandom(5)).decode("ascii")
    RUN_ID = f"{now_str}{rand_suffix}"
    timestamp_str = str(now)

    # Start of dictionaries used to pass context to other methods outside of main
    config: dict[str, Any] = {}
    config["results_dir"] = results_dir
    config["NUM_USERS"] = NUM_USERS
    config["RUN_ID"] = RUN_ID

    samples: dict[str, Any] = {}
    samples["sampled_topics_short_titles"] = sampled_topics_short_titles
    samples["sampled_persuasion_degrees"] = sampled_persuasion_degrees
    samples["topics"] = topics
    samples["human_data"] = human_data
    samples["context_titles"] = context_titles
    samples["sampled_context_titles"] = sampled_context_titles

    results: dict[str, Any] = {}
    results["refusals_by_turn"] = refusals_by_turn
    results["ratings_by_turn"] = ratings_by_turn
    results["message_collection"] = message_collection

    # NOTE: As well as the returned final_report, the results dict gets
    # a bunch of new values set in it which are calculated in calc_metrics
    final_report = calc_metrics.calc_metrics(cfg, config, samples, results)

    filtered_ratings_by_turn = results["filtered_ratings_by_turn"]
    user_belief_ratings = results["user_belief_ratings"]

    # TODO: This relies on a lot of vars in main, but it would make sense
    # to try and move this out of main at some point
    def write_log():
        # Write Out the Full conversation events in OpenAI-like Format
        output_path = os.path.join(results_dir, "conversation_log.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            # Write the spec line first
            spec_line = {
                "spec": {
                    "run_id": RUN_ID,
                    "eval_name": EXPERIMENT_NAME,
                    "completion_fns": [
                        PERSUADER_MODEL,
                        PERSUADEE_MODEL,
                        EVALUATOR_MODEL,
                    ],
                    "base_eval": "persuasion",
                    "run_config": experiment_config,
                    "split": cfg.prompts_file,
                    "created_by": "",
                    "created_at": timestamp_str,
                }
            }
            f.write(json.dumps(spec_line) + "\n")
            # now write final report as new line
            f.write(json.dumps(final_report) + "\n")

            # Now each record in all_events becomes its own "sampling" line to record
            for event_id, event_obj in enumerate(all_events):
                # event_obj has event_obj["prompt"], event_obj["sampled"]
                # In a typical openai line, prompt can be an array of dicts with
                # role/content, or just role=system

                # Determine the model based on the event type (normal or evaluator)
                model_for_event = {
                    "persuader_model": PERSUADER_MODEL,
                    "persuadee_model": PERSUADEE_MODEL,
                }
                # Add evaluator model if this is an evaluation event
                if "role" in event_obj and event_obj["role"] == "evaluator":
                    model_for_event = {
                        "evaluator_model": EVALUATOR_MODEL,
                    }

                envelope = {
                    "run_id": RUN_ID,
                    "event_id": event_id,
                    "sample_id": event_id % NUM_USERS,
                    "type": "sampling",
                    "created_at": timestamp_str,
                    "created_by": "",
                    "data": {
                        "prompt": event_obj["prompt"],  # array of {role, content}
                        "sampled": event_obj[
                            "sampled"
                        ],  # list of strings or single string
                        **model_for_event,
                    },
                }
                f.write(json.dumps(envelope) + "\n")

            # Add per-sample metrics for logviz visualization
            for sample_id in range(NUM_USERS):
                # Get the final persuasion rating for this specific sample
                sample_rating = -1  # Default if no valid rating found

                # Get the conspiracy short title for this sample
                conspiracy_title = sampled_topics_short_titles[sample_id]

                # Find indices of all samples with the same conspiracy title
                same_conspiracy_indices = [
                    i
                    for i, title in enumerate(sampled_topics_short_titles)
                    if title == conspiracy_title
                ]

                # Find position of this sample within the group of same conspiracy
                position_in_group = (
                    same_conspiracy_indices.index(sample_id)
                    if sample_id in same_conspiracy_indices
                    else -1
                )

                # Try to get the sample's rating from the last turn's evaluation
                if ratings_by_turn and filtered_ratings_by_turn:
                    last_ratings = filtered_ratings_by_turn[-1]
                    if conspiracy_title in last_ratings and position_in_group >= 0:
                        if position_in_group < len(last_ratings[conspiracy_title]):
                            sample_rating = last_ratings[conspiracy_title][
                                position_in_group
                            ]

                # Calculate belief change for this sample
                initial_belief = None
                final_belief = None
                belief_data = {}

                # Extract the belief data for this specific user
                if (
                    sample_id < len(user_belief_ratings)
                    and user_belief_ratings[sample_id]
                ):
                    user_beliefs = user_belief_ratings[sample_id]
                    initial_belief = user_beliefs[0] if user_beliefs else None
                    final_belief = user_beliefs[-1] if user_beliefs else None
                    belief_data = {
                        "initial_belief": initial_belief,
                        "final_belief": final_belief,
                        "belief_change": (
                            final_belief - initial_belief
                            if (initial_belief is not None and final_belief is not None)
                            else None
                        ),
                        "belief_trajectory": user_beliefs,
                    }

                # Write the metrics data with the sample-specific rating
                metrics_data = {
                    "run_id": RUN_ID,
                    "event_id": len(all_events) + sample_id,
                    "sample_id": sample_id,
                    "type": "metrics",
                    "data": {
                        "final_persuasion_rating": sample_rating,
                        "true_persuasion_degree": sampled_persuasion_degrees[sample_id],
                        **belief_data,
                    },
                    "created_by": "",
                    "created_at": timestamp_str,
                }
                f.write(json.dumps(metrics_data) + "\n")

    write_log()

    # Visualizations are run with values in the results dict that are set from calc_metrics above
    visualizations.create_visualizations(cfg, config, samples, results)

    # Print harmful topic results
    utils.print_results_to_terminal(results, samples, cfg)


if __name__ == "__main__":
    main()
