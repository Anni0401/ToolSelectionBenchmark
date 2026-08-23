"""
Retrieval Model Training Script

This script trains a Sentence Transformer model for retrieval using MultipleNegativesRankingLoss.
It uses data loading logic adapted from PORT project structure.
"""

import json
import logging
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, Set, List, Optional, Callable, Any, Tuple
import traceback
import pretty_errors
import random

import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from transformers import set_seed
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import (
    LoggingHandler,
    models,
    losses,
    InputExample,
    SentenceTransformer
)
from src_port.retrieval_evaluator import DeviceAwareInformationRetrievalEvaluator

# Set up environment
load_dotenv()
app_path = os.environ.get("APP_PATH", os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(app_path))

try:
    from src_port.dataset_helper import DatasetDownloader
except ImportError:
    logger.error("Failed to import DatasetDownloader. Ensure src_port is in the Python path.")
    class DatasetDownloader:
        def __init__(self, dataset_name):
            logger.error("Using dummy DatasetDownloader.")
            self.dataset_name = dataset_name
        def get_dataset(self):
            raise NotImplementedError("Dummy DatasetDownloader cannot load data.")
        def post_process_answers(self, dataset):
            logger.warning("Dummy DatasetDownloader cannot post-process.")
            return dataset

from src.utils.util_functions import ColoredFormatter

# Configure detailed logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_file = os.environ.get("LOG_FILE", "")
if log_file:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)


def create_api_triplets(dataset_split,
                        api_corpus: List[str],
                        num_negatives: int = 1,
                        random_neg: bool = True) -> List[Tuple[str, str, str]]:
    triplets = []
    corpus_set = set(api_corpus)
    skipped_count = 0

    logger.info(f"Creating triplets from dataset split with {len(dataset_split)} items.")
    logger.info(f"Using corpus of size {len(api_corpus)}.")

    for row in tqdm(dataset_split, desc="Creating API triplets"):
        query = row.get("query")
        positive_api = row.get("api_description")

        if not query or not positive_api:
            skipped_count += 1
            continue

        if positive_api not in corpus_set:
            logger.warning(f"Positive API not found in corpus: {positive_api[:100]}...")
            skipped_count += 1
            continue

        potential_negatives = list(corpus_set - {positive_api})

        if not potential_negatives:
            logger.warning(f"No potential negatives found for query: {query[:100]}...")
            skipped_count += 1
            continue

        num_to_sample = min(num_negatives, len(potential_negatives))

        if random_neg:
            selected_negatives = random.sample(potential_negatives, num_to_sample)
        else:
            selected_negatives = potential_negatives[:num_to_sample]

        for negative_api in selected_negatives:
            clean_query = str(query).strip()
            clean_pos = str(positive_api).strip()
            clean_neg = str(negative_api).strip()
            if clean_query and clean_pos and clean_neg:
                triplets.append((clean_query, clean_pos, clean_neg))
            else:
                logger.warning(f"Skipping triplet due to empty field after cleaning.")
                skipped_count += 1

    logger.info(f"Created {len(triplets)} triplets.")
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} rows due to missing data or lack of negatives.")

    if triplets:
        logger.info("Sample API triplets:")
        for i, triplet in enumerate(triplets[:3]):
            a, p, n = triplet
            logger.info(f"  Example {i+1}:")
            logger.info(f"    Anchor (Query): {a[:100]}{'...' if len(a) > 100 else ''}")
            logger.info(f"    Positive (API): {p[:100]}{'...' if len(p) > 100 else ''}")
            logger.info(f"    Negative (API): {n[:100]}{'...' if len(n) > 100 else ''}")

    return triplets


def log_to_wandb(score_dict: dict,
                 epoch: int,
                 steps: int) -> None:
    logger.info(f"Logging metrics to W&B at epoch {epoch}, step {steps}")
    things_to_score = {"epoch": epoch, "steps": steps}

    if not isinstance(score_dict, dict):
        things_to_score["score"] = float(score_dict)
        logger.info(f"  score: {float(score_dict)}")
    else:
        for main_key, main_value in score_dict.items():
            if isinstance(main_value, dict):
                for metric_key, metric_value in main_value.items():
                    if isinstance(metric_value, dict):
                        for k, v in metric_value.items():
                            log_key = f"{main_key}_{metric_key}_{k}"
                            if isinstance(v, (int, float, np.number)):
                                things_to_score[log_key] = float(v)
                                logger.info(f"  {log_key}: {float(v)}")
                    elif isinstance(metric_value, (int, float, np.number)):
                        log_key = f"{main_key}_{metric_key}"
                        things_to_score[log_key] = float(metric_value)
                        logger.info(f"  {log_key}: {float(metric_value)}")
            elif isinstance(main_value, (int, float, np.number)):
                things_to_score[main_key] = float(main_value)
                logger.info(f"  {main_key}: {float(main_value)}")

    wandb.log(things_to_score)


def create_api_evaluator(triplets: List[Tuple[str, str, str]],
                         eval_api_corpus: List[str],
                         batch_size: int = 16,
                         name: str = 'eval') -> DeviceAwareInformationRetrievalEvaluator:
    queries = {}
    corpus = {}
    relevant_docs = {}

    api_to_doc_id = {api_desc: f"doc_{i}" for i, api_desc in enumerate(eval_api_corpus)}
    corpus = {doc_id: api_desc for api_desc, doc_id in api_to_doc_id.items()}

    query_count = 0
    anchor_to_query_id = {}

    logger.info(f"Preparing evaluator '{name}' with {len(triplets)} triplets and {len(corpus)} corpus items.")

    for anchor, positive, _ in triplets:
        if anchor not in anchor_to_query_id:
            query_id = f"q_{query_count}"
            anchor_to_query_id[anchor] = query_id
            queries[query_id] = anchor
            relevant_docs[query_id] = set()
            query_count += 1
        else:
            query_id = anchor_to_query_id[anchor]

        if positive in api_to_doc_id:
            pos_doc_id = api_to_doc_id[positive]
            relevant_docs[query_id].add(pos_doc_id)
        else:
            logger.warning(f"Positive API description not found in eval corpus map for query '{anchor[:50]}...': '{positive[:50]}...'")

    if not queries:
        logger.error("No queries generated for the evaluator. Check input triplets and corpus.")
        raise ValueError("Cannot create evaluator with no queries.")
    if not corpus:
        logger.error("Corpus is empty. Cannot create evaluator.")
        raise ValueError("Cannot create evaluator with empty corpus.")
    if not relevant_docs:
        logger.warning("Relevant docs mapping is empty. Evaluator might not produce meaningful results.")

    evaluator = DeviceAwareInformationRetrievalEvaluator(
        queries,
        corpus,
        relevant_docs,
        batch_size=batch_size,
        name=name,
        show_progress_bar=(len(queries) > 10),
        ndcg_at_k=[1, 3, 5, 10],
        accuracy_at_k=[1, 3, 5, 10]
    )

    logger.info(f"Created evaluator '{name}' with {len(queries)} queries, {len(corpus)} corpus docs.")
    return evaluator


def evaluate_on_test(model,
                     test_triplets,
                     eval_api_corpus,
                     output_dir,
                     batch_size: int = 16) -> dict:
    logger.info(f"Evaluating model on {len(test_triplets)} test triplets")

    if not test_triplets:
        logger.warning("No test triplets available for evaluation")
        return {}
    if not eval_api_corpus:
        logger.error("Evaluation API corpus is empty. Cannot perform test evaluation.")
        return {}

    try:
        test_evaluator = create_api_evaluator(test_triplets,
                                              eval_api_corpus,
                                              batch_size=batch_size,
                                              name='test')
    except ValueError as e:
        logger.error(f"Failed to create test evaluator: {e}")
        return {}

    logger.info("Running test evaluation...")
    eval_start = time.time()
    test_stats = test_evaluator(model, output_path=os.path.join(output_dir, "test_results"))
    eval_end = time.time()
    eval_time = eval_end - eval_start

    logger.info(f"Test evaluation completed in {eval_time:.2f} seconds")
    logger.info(f"Test evaluation results:")
    for main_key, main_value in test_stats.items():
        if isinstance(main_value, dict):
            for metric_key, metric_value in main_value.items():
                logger.info(f"  {main_key}_{metric_key}: {metric_value}")
        else:
            logger.info(f"  {main_key}: {main_value}")

    test_eval_path = os.path.join(output_dir, "test_eval_results.json")
    try:
        with open(test_eval_path, 'w', encoding='utf-8') as f_out:
            def default_serializer(o):
                if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8,
                                  np.uint16, np.uint32, np.uint64)):
                    return int(o)
                elif isinstance(o, (np.float_, np.float16, np.float32,
                                    np.float64)):
                    return float(o)
                elif isinstance(o, (np.ndarray,)):
                    return o.tolist()
                elif isinstance(o, set):
                    return list(o)
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

            json.dump(test_stats, f_out, ensure_ascii=False, indent=2, default=default_serializer)
        logger.info(f"Saved test evaluation results to {test_eval_path}")
    except Exception as e:
        logger.error(f"Failed to save test evaluation results to {test_eval_path}: {e}")

    return test_stats


def main(args):
    script_start_time = time.time()
    logger.info(f"Starting API retrieval model training with args: {args}")

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model_name = args.model_name
        train_batch_size = args.train_batch_size
        retriever_max_seq_length = args.retriever_max_seq_length
        num_epochs = args.epochs
        model_save_path = args.output_dir

        logger.info(f"Model configuration:")
        logger.info(f"  Model name/path: {model_name}")
        logger.info(f"  Batch size: {train_batch_size}")
        logger.info(f"  Max sequence length: {retriever_max_seq_length}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Output directory: {model_save_path}")
        logger.info(f"  Pooling: {args.pooling}")

        os.makedirs(model_save_path, exist_ok=True)
        logger.info(f"Ensured output directory exists: {model_save_path}")

        config_path = os.path.join(model_save_path, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Saved configuration to {config_path}")

        logger.info(f"Loading retriever model {model_name}...")
        model_load_start = time.time()

        try:
            is_path = os.path.isdir(model_name)

            if args.use_pre_trained_model and not is_path:
                logger.info("Loading pretrained SentenceTransformer model from Hugging Face.")
                model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
            elif is_path:
                logger.info(f"Loading SentenceTransformer model from path: {model_name}")
                model = SentenceTransformer(model_name, device=device)
            else:
                logger.info("Creating new SentenceTransformer model from base transformer.")
                word_embedding_model = models.Transformer(model_name,
                                                          max_seq_length=retriever_max_seq_length,
                                                          tokenizer_args={"trust_remote_code": True},
                                                          model_args={"trust_remote_code": True},
                                                          config_args={"trust_remote_code": True})
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
                model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

            logger.info(f"Model loaded successfully in {time.time() - model_load_start:.2f} seconds")
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model has {num_params:,} parameters")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise

        logger.info(f"Loading dataset: {args.dataset}")
        data_load_start = time.time()
        try:
            downloader = DatasetDownloader(dataset_name=args.dataset)
            dataset = downloader.get_dataset()
            logger.info(f"Dataset loaded in {time.time() - data_load_start:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load dataset '{args.dataset}': {e}")
            logger.error(traceback.format_exc())
            raise

        prep_start = time.time()

        train_split_name = 'train'
        dev_split_name = 'validation' if 'validation' in dataset else 'test'
        test_split_name = 'test'

        if train_split_name not in dataset:
            raise ValueError(f"Training split '{train_split_name}' not found in dataset.")
        if dev_split_name not in dataset:
            raise ValueError(f"Development split '{dev_split_name}' not found in dataset.")
        if args.evaluate_on_test and test_split_name not in dataset:
            logger.warning(f"Test split '{test_split_name}' not found, disabling test evaluation.")
            args.evaluate_on_test = False

        if args.max_train_samples and args.max_train_samples < len(dataset[train_split_name]):
            logger.info(f"Sampling {args.max_train_samples} instances from train split.")
            indices = random.sample(range(len(dataset[train_split_name])), args.max_train_samples)
            dataset[train_split_name] = dataset[train_split_name].select(indices)

        max_dev_samples = args.max_train_samples
        if max_dev_samples and max_dev_samples < len(dataset[dev_split_name]):
            logger.info(f"Sampling {max_dev_samples} instances from dev split '{dev_split_name}'.")
            indices = random.sample(range(len(dataset[dev_split_name])), max_dev_samples)
            dataset[dev_split_name] = dataset[dev_split_name].select(indices)

        if args.evaluate_on_test:
            max_test_samples = args.max_train_samples
            if max_test_samples and max_test_samples < len(dataset[test_split_name]):
                logger.info(f"Sampling {max_test_samples} instances from test split '{test_split_name}'.")
                indices = random.sample(range(len(dataset[test_split_name])), max_test_samples)
                dataset[test_split_name] = dataset[test_split_name].select(indices)

        logger.info("Creating API corpora...")
        required_field = 'api_description'
        if required_field not in dataset[train_split_name].column_names:
            raise ValueError(f"Required field '{required_field}' not found in train split.")
        if required_field not in dataset[dev_split_name].column_names:
            raise ValueError(f"Required field '{required_field}' not found in dev split '{dev_split_name}'.")

        train_api_corpus = list(set(dataset[train_split_name][required_field]))
        eval_corpus_set = set(dataset[dev_split_name][required_field])
        if args.evaluate_on_test:
            if required_field not in dataset[test_split_name].column_names:
                raise ValueError(f"Required field '{required_field}' not found in test split '{test_split_name}'.")
            eval_corpus_set.update(dataset[test_split_name][required_field])
        eval_api_corpus = list(eval_corpus_set)

        logger.info(f"Train corpus size: {len(train_api_corpus)}")
        logger.info(f"Eval corpus size: {len(eval_api_corpus)}")
        if not train_api_corpus or not eval_api_corpus:
            raise ValueError("API corpora are empty. Check dataset content.")

        logger.info("Creating triplets...")
        train_triplets = create_api_triplets(dataset[train_split_name],
                                             train_api_corpus,
                                             args.negatives_per_sample,
                                             args.random_negatives)
        dev_triplets = create_api_triplets(dataset[dev_split_name],
                                           eval_api_corpus,
                                           args.negatives_per_sample,
                                           args.random_negatives)

        test_triplets = []
        if args.evaluate_on_test:
            test_triplets = create_api_triplets(dataset[test_split_name],
                                                eval_api_corpus,
                                                args.negatives_per_sample,
                                                args.random_negatives)
            logger.info(f"Test triplets: {len(test_triplets)}")

        logger.info(f"Train triplets: {len(train_triplets)}")
        logger.info(f"Dev triplets: {len(dev_triplets)}")

        if not train_triplets:
            logger.error("No training triplets were generated. Cannot proceed.")
            return
        if not dev_triplets:
            logger.warning("No development triplets were generated. Evaluation might fail or be meaningless.")

        logger.info(f"Creating training dataset and dataloader [{len(train_triplets)} triplets]")
        train_examples = [InputExample(texts=[q, p, n]) for q, p, n in train_triplets]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
        logger.info(f"Created training dataloader with {len(train_dataloader)} batches")

        logger.info("Creating MultipleNegativesRankingLoss")
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        logger.info("Creating evaluation setup")
        if dev_triplets and eval_api_corpus:
            try:
                ir_evaluator = create_api_evaluator(dev_triplets,
                                                    eval_api_corpus,
                                                    batch_size=args.eval_batch_size,
                                                    name='dev')
            except ValueError as e:
                logger.error(f"Failed to create development evaluator: {e}. Proceeding without evaluation.")
                ir_evaluator = None
        else:
            logger.warning("Cannot create development evaluator due to missing triplets or corpus.")
            ir_evaluator = None

        logger.info(f"Data preparation finished in {time.time() - prep_start:.2f} seconds.")

        # Calculate evaluation steps based on fraction if provided
        evaluation_step_interval = 0 # Default: evaluate per epoch
        steps_per_epoch = len(train_dataloader)
        if ir_evaluator and args.eval_steps is not None and args.eval_steps > 0:
             if not (0 < args.eval_steps <= 1):
                 raise ValueError("eval_steps must be a float between 0 (exclusive) and 1 (inclusive)")
             evaluation_step_interval = max(1, int(steps_per_epoch * args.eval_steps))
             logger.info(f"Evaluation strategy: 'steps'. Evaluating every {evaluation_step_interval} steps ({args.eval_steps*100:.2f}% of {steps_per_epoch} steps per epoch).")
        elif ir_evaluator:
             logger.info(f"Evaluation strategy: 'epoch'. Evaluating every {steps_per_epoch} steps (at the end of each epoch).")
             evaluation_step_interval = steps_per_epoch # Evaluate at the end of epoch if eval_steps <= 0 or None
        else:
             logger.info("No evaluator available, skipping evaluation steps calculation.")


        wandb_run = None
        if not args.do_eval_only and not args.disable_wandb:
            run_name = f"SentTrans-{args.model_name.split('/')[-1]}-{args.dataset}-{datetime.now().strftime('%Y%m%d-%H%M')}"
            logger.info(f"Initializing W&B with run name: {run_name}")
            try:
                wandb_run = wandb.init(project="API-Retriever", name=run_name)
                wandb.config.update(args)
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                wandb_run = None

        if ir_evaluator:
            logger.info("Evaluating model before training...")
            eval_start = time.time()
            stats = ir_evaluator(model, output_path=model_save_path)
            eval_end = time.time()
            eval_time = eval_end - eval_start

            logger.info(f"Initial evaluation completed in {eval_time:.2f} seconds")
            logger.info(f"Initial evaluation results:")
            for main_key, main_value in stats.items():
                if isinstance(main_value, dict):
                    for metric_key, metric_value in main_value.items():
                        logger.info(f"  {main_key}_{metric_key}: {metric_value}")
                else:
                    logger.info(f"  {main_key}: {main_value}")

            initial_eval_path = os.path.join(model_save_path, "initial_eval_results.json")
            try:
                with open(initial_eval_path, 'w', encoding='utf-8') as f_out:
                    def default_serializer(o):
                        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                                          np.int16, np.int32, np.int64, np.uint8,
                                          np.uint16, np.uint32, np.uint64)):
                            return int(o)
                        elif isinstance(o, (np.float_, np.float16, np.float32,
                                            np.float64)):
                            return float(o)
                        elif isinstance(o, (np.ndarray,)):
                            return o.tolist()
                        elif isinstance(o, set):
                            return list(o)
                        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
                    json.dump(stats, f_out, ensure_ascii=False, indent=2, default=default_serializer)
                logger.info(f"Saved initial evaluation results to {initial_eval_path}")
            except Exception as e:
                logger.error(f"Failed to save initial evaluation results: {e}")

            if wandb_run:
                log_to_wandb({"initial_" + k: v for k, v in stats.items()}, epoch=0, steps=0)
        else:
            logger.warning("Skipping initial evaluation as evaluator could not be created.")

        if not args.do_eval_only:
            logger.info("Starting training...")
            train_start = time.time()

            callback_func = log_to_wandb if wandb_run else None

            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                warmup_steps=args.warmup_steps,
                evaluator=ir_evaluator,
                evaluation_steps=evaluation_step_interval, # Use calculated interval
                callback=callback_func,
                use_amp=True,
                checkpoint_path=model_save_path,
                checkpoint_save_total_limit=args.checkpoint_save_total_limit,
                checkpoint_save_steps=evaluation_step_interval, # Use calculated interval for checkpoints too
                optimizer_params={"lr": args.lr},
                scheduler=args.scheduler,
                save_best_model=True if ir_evaluator else False,
                max_grad_norm=args.max_grad_norm,
                weight_decay=args.weight_decay
            )

            train_end = time.time()
            train_duration = train_end - train_start
            logger.info(f"Training completed in {train_duration:.2f} seconds "
                        f"({str(timedelta(seconds=int(train_duration)))})")

            final_model_path = os.path.join(model_save_path, "final")
            logger.info(f"Saving final model to {final_model_path}")
            model.save(final_model_path)
            logger.info(f"Model saved successfully")

            if args.push_to_hub and os.environ.get('HF_TOKEN'):
                repo_name = args.hub_repo_name or f"api-retriever-{args.model_name.split('/')[-1]}-{args.dataset}"
                logger.info(f"Pushing model to Hugging Face Hub: {repo_name}")
                try:
                    model.save_to_hub(repo_id=repo_name,
                                      commit_message="Add final trained model",
                                      private=not args.public_model,
                                      exist_ok=True)
                    logger.info(f"Model pushed successfully to Hugging Face Hub: {repo_name}")
                except Exception as e:
                    logger.error(f"Failed to push model to hub: {e}")
                    logger.error(traceback.format_exc())

            if wandb_run:
                logger.info("Finishing W&B run")
                wandb.finish()
                wandb_run = None

        if args.evaluate_on_test and test_triplets:
            logger.info("Evaluating model on test set...")

            eval_model = None
            if args.do_eval_only and os.path.isdir(args.model_name):
                logger.info(f"Loading model from {args.model_name} for test evaluation")
                try:
                    eval_model = SentenceTransformer(args.model_name, device=device)
                except Exception as e:
                    logger.error(f"Error loading model for test evaluation: {e}")
                    logger.error(traceback.format_exc())
                    eval_model = None
            elif not args.do_eval_only:
                best_model_path = model_save_path
                if os.path.exists(best_model_path):
                    logger.info(f"Loading best model from {best_model_path} for test evaluation")
                    try:
                        eval_model = SentenceTransformer(best_model_path, device=device)
                    except Exception as e:
                        logger.warning(f"Error loading best model from {best_model_path}, using final model instead: {e}")
                        eval_model = model
                else:
                    logger.warning(f"Best model path {best_model_path} not found, using final model.")
                    eval_model = model
            else:
                logger.error(f"Evaluation only requested, but model_name '{args.model_name}' is not a valid directory path.")
                eval_model = None

            if eval_model:
                test_stats = evaluate_on_test(eval_model, test_triplets, eval_api_corpus, model_save_path, batch_size=args.eval_batch_size)

                if args.do_eval_only and not args.disable_wandb and not wandb_run:
                    run_name = f"Test-{args.model_name.split('/')[-1]}-{args.dataset}-{datetime.now().strftime('%Y%m%d-%H%M')}"
                    logger.info(f"Initializing W&B for test results with run name: {run_name}")
                    try:
                        wandb_run = wandb.init(project="API-Retriever-Test", name=run_name)
                        wandb.config.update(args)
                        log_to_wandb({"test_" + k: v for k, v in test_stats.items()}, epoch=0, steps=0)
                        wandb.finish()
                        wandb_run = None
                    except Exception as e:
                        logger.error(f"Failed to initialize or log test results to W&B: {e}")
            else:
                logger.error("Could not load a model for test evaluation.")

        script_end_time = time.time()
        total_time = script_end_time - script_start_time
        logger.info(f"Script execution completed in {total_time:.2f} seconds "
                    f"({str(timedelta(seconds=int(total_time)))})")

    except Exception as e:
        logger.error(f"Unhandled error during execution: {e}")
        logger.error(traceback.format_exc())
        if wandb_run:
            wandb.finish()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an API retrieval model")

    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name for training and evaluation (e.g., "bfcl", "apibank", "toolbench")')
    parser.add_argument("--random_negatives", action="store_true", default=True,
                        help="Randomly select negatives (default: True)")
    parser.add_argument("--no_random_negatives", action="store_false", dest="random_negatives",
                        help="Select negatives sequentially instead of randomly")
    parser.add_argument("--negatives_per_sample", default=1, type=int,
                        help="Number of negative examples to use per positive sample")
    parser.add_argument("--max_train_samples", default=None, type=int,
                        help="Maximum number of training samples (if specified, randomly selects a subset)")
    parser.add_argument("--evaluate_on_test", action="store_true",
                        help="Evaluate the model on the test set after training")

    parser.add_argument("--model_name", required=True, type=str,
                        help="Transformer model name or path (e.g., 'bert-base-uncased', './my_model')")
    parser.add_argument("--retriever_max_seq_length", default=256, type=int,
                        help="Maximum sequence length for the retriever model")
    parser.add_argument("--pooling", default="mean", type=str, choices=["mean", "cls", "max"],
                        help="Pooling strategy (mean, cls, max)")
    parser.add_argument("--use_pre_trained_model", action="store_true",
                        help="Load model_name directly as a SentenceTransformer (if it's an SBERT model ID), otherwise build from base.")

    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Evaluation batch size")
    parser.add_argument("--preprocessing_batch_size", default=16, type=int,
                        help="Batch size for preprocessing steps like embedding corpus (if applicable)")
    parser.add_argument("--epochs", default=3, type=int,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Number of warmup steps")
    parser.add_argument("--lr", default=2e-5, type=float,
                        help="Learning rate")
    parser.add_argument("--eval_steps", default=None, type=float,
                        help="Evaluate model every X fraction of steps within an epoch (e.g., 0.1 for every 10%). If None or <= 0, evaluate per epoch.")
    parser.add_argument("--do_eval_only", action="store_true",
                        help="Only run evaluation, no training (requires model_name to be a path to a trained model)")
    parser.add_argument("--output_dir", default="output/api_retriever",
                        help="Directory to save model checkpoints and results")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm for gradient clipping")
    parser.add_argument("--scheduler", default="warmupcosine", type=str, choices=["warmupcosine", "warmuplinear", "constantlr"],
                        help="Learning rate scheduler")
    parser.add_argument("--checkpoint_save_total_limit", default=2, type=int,
                        help="Maximum number of checkpoints to keep (SBERT saves best based on evaluator)")

    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push trained model to Hugging Face Hub")
    parser.add_argument("--hub_repo_name", type=str, default=None,
                        help="Repository name for Hugging Face Hub (optional)")
    parser.add_argument("--public_model", action="store_true",
                        help="Make the pushed model publicly available")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file (if not set, only console logging is used)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    set_seed(args.seed)

    if args.log_file:
        os.environ["LOG_FILE"] = args.log_file
        log_file = args.log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    main(args)