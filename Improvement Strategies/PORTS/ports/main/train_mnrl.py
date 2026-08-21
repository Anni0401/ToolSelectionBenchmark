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
import math

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
from src.port.retrieval_evaluator import DeviceAwareInformationRetrievalEvaluator

# Set up environment
load_dotenv()
app_path = os.environ.get("APP_PATH", os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(app_path))


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

try:
    from src.port.dataset_helper import DatasetDownloader
except ImportError:
    logger.error("Failed to import DatasetDownloader. Ensure src.port is in the Python path.")
    class DatasetDownloader:
        def __init__(self, dataset_name):
            logger.error("Using dummy DatasetDownloader.")
            self.dataset_name = dataset_name
        def get_dataset(self):
            raise NotImplementedError("Dummy DatasetDownloader cannot load data.")
        def post_process_answers(self, dataset):
            logger.warning("Dummy DatasetDownloader cannot post-process.")
            return dataset


def create_api_triplets(dataset_split,
                        api_corpus: List[str],
                        num_negatives: int = 1,
                        random_neg: bool = True) -> List[Tuple[str, str, str]]:
    triplets = []
    corpus_set = set(api_corpus)
    skipped_count = 0
    
    # First, build a mapping of queries to their positive API descriptions
    query_to_positives = {}
    for row in dataset_split:
        query = row.get("query_for_retrieval")
        positive_api = row.get("api_description")
        
        if query and positive_api:
            if query not in query_to_positives:
                query_to_positives[query] = set()
            query_to_positives[query].add(positive_api)
    
    logger.info(f"Found {len(query_to_positives)} unique queries with positive examples")
    logger.info(f"Creating triplets from dataset ({len(dataset_split)} examples)")
    
    for row in tqdm(dataset_split, desc="Creating triplets", mininterval=2.0):
        query = row.get("query_for_retrieval")
        positive_api = row.get("api_description")

        if not query or not positive_api:
            skipped_count += 1
            continue

        if positive_api not in corpus_set:
            skipped_count += 1
            continue
        
        # Get all positives for this query and exclude them from potential negatives
        query_positives = query_to_positives.get(query, {positive_api})
        potential_negatives = list(corpus_set - query_positives)

        if not potential_negatives:
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
                skipped_count += 1

    logger.info(f"Created {len(triplets)} triplets, skipped {skipped_count} rows")

    if triplets:
        a, p, n = triplets[0]
        logger.info(f"Sample triplet: Q: '{a[:50]}...', API: '{p[:50]}...', Neg: '{n[:50]}...'")

    return triplets


def log_to_wandb(score_dict: dict,
                 epoch: int,
                 steps: int,
                 prefix: str = "eval") -> None:
    """Logs metrics to W&B with structured keys."""
    if wandb.run is None:
        logger.warning("wandb.init() has not been called. Skipping wandb.log().")
        return

    log_prefix = f"{prefix}/" if prefix else ""
    logger.info(f"Logging {prefix} metrics to W&B at epoch {epoch}, step {steps}")

    flat_scores = {}

    def flatten_dict(d, parent_key='', sep='/'):
        items = []
        for k, v in d.items():
            sanitized_k = str(k).replace('@', '_at_')
            new_key = f"{parent_key}{sep}{sanitized_k}" if parent_key else sanitized_k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, np.number)):
                items.append((new_key, float(v)))
        return dict(items)

    if isinstance(score_dict, dict):
        flat_scores = flatten_dict(score_dict)
    elif isinstance(score_dict, (int, float, np.number)):
        flat_scores = {"score": float(score_dict)}
    else:
        logger.warning(f"Unsupported score_dict type for logging: {type(score_dict)}")
        return

    wandb_log_data = {f"{log_prefix}{k}": v for k, v in flat_scores.items()}
    wandb_log_data[f"{prefix}/epoch"] = epoch
    wandb_log_data[f"{prefix}/step"] = steps

    key_metrics = {k: v for k, v in flat_scores.items() if 'accuracy_at_1' in k or 'ndcg_at_1' in k}
    for key, value in key_metrics.items():
        logger.info(f"  {log_prefix}{key}: {value}")
        
    if len(key_metrics) < 3:
        additional_count = 0
        for k, v in flat_scores.items():
            if k not in key_metrics and isinstance(v, (int, float)) and additional_count < 3:
                logger.info(f"  {log_prefix}{k}: {v}")
                additional_count += 1

    wandb.log(wandb_log_data, step=steps)


def create_api_evaluator(triplets: List[Tuple[str, str, str]],
                         eval_api_corpus: List[str],
                         batch_size: int = 16,
                         name: str = 'eval',
                         k_values_accuracy: list = [1, 3, 5, 10],
                         k_values_ndcg: list = [1, 3, 5, 10]) -> DeviceAwareInformationRetrievalEvaluator:
    queries = {}
    corpus = {}
    relevant_docs = {}

    api_to_doc_id = {api_desc: f"doc_{i}" for i, api_desc in enumerate(eval_api_corpus)}
    corpus = {doc_id: api_desc for api_desc, doc_id in api_to_doc_id.items()}

    query_count = 0
    anchor_to_query_id = {}
    warning_count = 0

    logger.info(f"Creating evaluator '{name}' with {len(triplets)} triplets")

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
            warning_count += 1
            if warning_count <= 5:
                logger.warning(f"Missing API (showing 5/{warning_count}): '{positive[:30]}...'")
            if warning_count == 6:
                logger.warning(f"Additional missing APIs found. Suppressing further warnings.")

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
        ndcg_at_k=k_values_ndcg,
        accuracy_at_k=k_values_accuracy
    )

    logger.info(f"Evaluator created with {len(queries)} queries, {len(corpus)} corpus items")
    return evaluator


def evaluate_on_test(model,
                     test_triplets,
                     eval_api_corpus,
                     output_dir,
                     batch_size: int = 16,
                     k_values_accuracy: list = [1, 3, 5, 10],
                     k_values_ndcg: list = [1, 3, 5, 10],
                     epoch: int = -1,
                     steps: int = -1) -> dict:
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
                                              name='test',
                                              k_values_accuracy=k_values_accuracy,
                                              k_values_ndcg=k_values_ndcg)
    except ValueError as e:
        logger.error(f"Failed to create test evaluator: {e}")
        return {}

    logger.info("Running test evaluation...")
    eval_start = time.time()
    test_stats = test_evaluator(model, output_path=os.path.join(output_dir, "test_results"))
    eval_end = time.time()
    eval_time = eval_end - eval_start

    logger.info(f"Test evaluation completed in {eval_time:.2f}s. Key results:")
    
    key_metrics = {}
    for main_key, main_value in test_stats.items():
        if isinstance(main_value, dict):
            for metric_key, metric_value in main_value.items():
                if 'accuracy@1' in metric_key or 'ndcg@1' in metric_key:
                    key_metrics[f"{main_key}_{metric_key}"] = metric_value
        elif main_key in ['map', 'mrr']:
            key_metrics[main_key] = main_value
    
    for key, value in key_metrics.items():
        logger.info(f"  {key}: {value}")

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

        logger.info(f"Model: {model_name}, BS: {train_batch_size}, Epochs: {num_epochs}, Pooling: {args.pooling}")

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
                model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
            elif is_path:
                model = SentenceTransformer(model_name, device=device)
            else:
                word_embedding_model = models.Transformer(model_name,
                                                          max_seq_length=retriever_max_seq_length,
                                                          tokenizer_args={"trust_remote_code": True},
                                                          model_args={"trust_remote_code": True},
                                                          config_args={"trust_remote_code": True})
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
                normalize = models.Normalize()
                model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize], device=device)

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
            indices = random.sample(range(len(dataset[train_split_name])), args.max_train_samples)
            dataset[train_split_name] = dataset[train_split_name].select(indices)

        max_dev_samples = args.max_train_samples
        if max_dev_samples and max_dev_samples < len(dataset[dev_split_name]):
            indices = random.sample(range(len(dataset[dev_split_name])), max_dev_samples)
            dataset[dev_split_name] = dataset[dev_split_name].select(indices)

        if args.evaluate_on_test:
            max_test_samples = args.max_train_samples
            if max_test_samples and max_test_samples < len(dataset[test_split_name]):
                indices = random.sample(range(len(dataset[test_split_name])), max_test_samples)
                dataset[test_split_name] = dataset[test_split_name].select(indices)

        logger.info(f"Dataset '{args.dataset}' loaded: {list(dataset.keys())}")

        logger.info("Creating API corpora...")
        required_field = 'api_description'
        if required_field not in dataset[train_split_name].column_names:
            raise ValueError(f"Required field '{required_field}' not found in train split.")
        if required_field not in dataset[dev_split_name].column_names:
            raise ValueError(f"Required field '{required_field}' not found in dev split '{dev_split_name}'.")

        # Define train API corpus from training split
        train_api_corpus = list(set(dataset[train_split_name][required_field]))
        logger.info(f"Train API corpus size: {len(train_api_corpus)}")

        # Define dev API corpus strictly from development split (for initial evaluation)
        dev_api_corpus = list(set(dataset[dev_split_name][required_field]))
        logger.info(f"Development API corpus size: {len(dev_api_corpus)}")

        # For test evaluation, create a combined corpus of train + test APIs
        test_api_corpus = []
        
        if args.evaluate_on_test:
            if required_field not in dataset[test_split_name].column_names:
                raise ValueError(f"Required field '{required_field}' not found in test split '{test_split_name}'.")
            
            logger.info(f"Final test evaluation will use dev_api_corpus (size: {len(dev_api_corpus)})")


        # Create dev triplets using only dev APIs for the corpus
        logger.info("Creating dev triplets...")
        dev_triplets = create_api_triplets(dataset[dev_split_name],
                                           dev_api_corpus,
                                           args.negatives_per_sample,
                                           args.random_negatives)
        logger.info(f"Dev triplets: {len(dev_triplets)}")

        # Create train triplets using train API corpus
        logger.info("Creating train triplets...")
        train_triplets = create_api_triplets(dataset[train_split_name],
                                             train_api_corpus,
                                             args.negatives_per_sample,
                                             args.random_negatives)
        logger.info(f"Train triplets: {len(train_triplets)}")

        # Create test triplets using dev corpus for negative sampling
        test_triplets = []
        if args.evaluate_on_test:
            logger.info("Creating test triplets with dev_api_corpus for negative sampling...")
            test_triplets = create_api_triplets(dataset[test_split_name],
                                                dev_api_corpus, # Use dev_api_corpus for negatives
                                                args.negatives_per_sample,
                                                args.random_negatives)
            logger.info(f"Test triplets: {len(test_triplets)}")

        if not train_triplets:
            logger.error("No training triplets were generated. Cannot proceed.")
            return
        if not dev_triplets:
            logger.warning("No development triplets were generated. Evaluation might fail or be meaningless.")

        train_examples = [InputExample(texts=[q, p, n]) for q, p, n in train_triplets]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)

        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        k_values_accuracy = getattr(args, 'k_eval_values_accuracy', [1, 3, 5, 10])
        k_values_ndcg = getattr(args, 'k_eval_values_ndcg', [1, 3, 5, 10])

        # For initial evaluation, use the dev API corpus
        if dev_triplets and dev_api_corpus:
            try:
                ir_evaluator = create_api_evaluator(dev_triplets,
                                                    dev_api_corpus,
                                                    batch_size=args.eval_batch_size,
                                                    name='dev',
                                                    k_values_accuracy=k_values_accuracy,
                                                    k_values_ndcg=k_values_ndcg)
            except ValueError as e:
                logger.error(f"Failed to create development evaluator: {e}. Proceeding without evaluation.")
                ir_evaluator = None
        else:
            logger.warning("Cannot create development evaluator due to missing triplets or corpus.")
            ir_evaluator = None

        evaluation_step_interval = 0
        steps_per_epoch = len(train_dataloader)
        num_training_steps = steps_per_epoch * num_epochs
        num_warmup_steps = math.ceil(num_training_steps * args.warmup_ratio)
        logger.info(f"Training config: Steps: {num_training_steps}, Warmup: {num_warmup_steps}, Eval every: {evaluation_step_interval}")

        if ir_evaluator and args.eval_steps is not None and args.eval_steps > 0:
             if not (0 < args.eval_steps <= 1):
                 raise ValueError("eval_steps must be a float between 0 (exclusive) and 1 (inclusive)")
             evaluation_step_interval = max(1, int(steps_per_epoch * args.eval_steps))
        elif ir_evaluator:
             evaluation_step_interval = steps_per_epoch
        else:
             logger.info("No evaluator available, skipping evaluation steps calculation.")

        wandb_run = None
        if not args.do_eval_only and not args.disable_wandb:
            # Use configurable project and run names
            project_name = args.wandb_project_name or "API-Retriever"
            run_name = args.wandb_run_name or f"SentTrans-{args.model_name.split('/')[-1]}-{args.dataset}-{datetime.now().strftime('%Y%m%d-%H%M')}"

            logger.info(f"Initializing W&B with project: {project_name}, run name: {run_name}")
            try:
                wandb_run = wandb.init(project=project_name, name=run_name)
                wandb.config.update(vars(args))
                # Watch the model if a logging frequency is set
                if getattr(args, "wandb_log_freq", 0) > 0:
                    wandb.watch(model, log_freq=args.wandb_log_freq)
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                wandb_run = None

        if ir_evaluator:
            eval_start = time.time()
            stats = ir_evaluator(model, output_path=model_save_path)
            eval_end = time.time()
            eval_time = eval_end - eval_start

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
            except Exception as e:
                logger.error(f"Failed to save initial evaluation results: {e}")

            if wandb_run:
                log_to_wandb({"initial_" + k: v for k, v in stats.items()}, epoch=0, steps=0)

        if not args.do_eval_only:
            train_start = time.time()

            def wandb_callback(score, epoch, steps):
                if hasattr(ir_evaluator, 'latest_scores') and ir_evaluator.latest_scores:
                     log_to_wandb(ir_evaluator.latest_scores, epoch=epoch, steps=steps, prefix="eval")
                else:
                     log_to_wandb({"score": score}, epoch=epoch, steps=steps, prefix="eval")

            callback_func = wandb_callback if wandb_run and ir_evaluator else None

            checkpoint_params = {}
            if args.save_checkpoints:
                checkpoint_params = {
                    "checkpoint_path": model_save_path,
                    "checkpoint_save_total_limit": args.checkpoint_save_total_limit,
                    "checkpoint_save_steps": evaluation_step_interval,
                    "save_best_model": True if ir_evaluator else False
                }

            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                warmup_steps=num_warmup_steps,
                evaluator=ir_evaluator,
                evaluation_steps=evaluation_step_interval,
                callback=callback_func,
                use_amp=True,
                optimizer_params={"lr": args.lr},
                scheduler=args.scheduler,
                max_grad_norm=args.max_grad_norm,
                weight_decay=args.weight_decay,
                **checkpoint_params
            )

            train_end = time.time()
            train_duration = train_end - train_start
            logger.info(f"Training completed in {train_duration:.2f}s ({str(timedelta(seconds=int(train_duration)))})")

            if args.save_checkpoints:
                final_model_path = os.path.join(model_save_path, "final")
                model.save(final_model_path)
                logger.info(f"Saved final model to {final_model_path}")

            # if args.push_to_hub and os.environ.get('HF_TOKEN'):
            #     repo_name = args.hub_repo_name or f"api-retriever-{args.model_name.split('/')[-1]}-{args.dataset}"
            #     try:
            #         model.save_to_hub(repo_id=repo_name,
            #                           commit_message="Add final trained model",
            #                           private=not args.public_model,
            #                           exist_ok=True)
            #     except Exception as e:
            #         logger.error(f"Failed to push model to hub: {e}")

            if wandb_run:
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
                # For final test evaluation, use the dev API corpus
                logger.info(f"Evaluating on test queries using dev_api_corpus (size: {len(dev_api_corpus)}) as the search space.")
                test_stats = evaluate_on_test(eval_model, 
                                              test_triplets, 
                                              dev_api_corpus,  # Use dev_api_corpus as the evaluation corpus
                                              model_save_path, 
                                              batch_size=args.eval_batch_size,
                                              k_values_accuracy=k_values_accuracy,
                                              k_values_ndcg=k_values_ndcg,
                                              epoch=-1,
                                              steps=-1)

                if args.do_eval_only and not args.disable_wandb and not wandb_run:
                    run_name = f"Test-{args.model_name.split('/')[-1]}-{args.dataset}-{datetime.now().strftime('%Y%m%d-%H%M')}"
                    try:
                        wandb_run = wandb.init(project="API-Retriever-Test", name=run_name)
                        wandb.config.update(args)
                        log_to_wandb({"test_" + k: v for k, v in test_stats.items()}, epoch=0, steps=0)
                        wandb.finish()
                        wandb_run = None
                    except Exception as e:
                        logger.error(f"Failed to initialize or log test results to W&B: {e}")

        script_end_time = time.time()
        total_time = script_end_time - script_start_time
        logger.info(f"Script execution completed in {total_time:.2f} seconds ({str(timedelta(seconds=int(total_time)))})")

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
    parser.add_argument("--pooling", default="cls", type=str, choices=["mean", "cls", "max"],
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
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Fraction of total training steps for warmup (0.0 to 1.0)")
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
    parser.add_argument("--save_checkpoints", action="store_true", default=False,
                        help="Whether to save model checkpoints during training")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file (if not set, only console logging is used)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    parser.add_argument("--k_eval_values_accuracy", nargs="+", type=int, default=[1, 3, 5, 10],
                        help="Values of k for accuracy@k evaluation metrics")
    parser.add_argument("--k_eval_values_ndcg", nargs="+", type=int, default=[1, 3, 5, 10],
                        help="Values of k for ndcg@k evaluation metrics")

    # Wandb configuration parameters
    parser.add_argument("--wandb_project_name", type=str, default=None,
                        help="W&B project name (default: 'API-Retriever')")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated based on model and dataset)")
    parser.add_argument("--wandb_log_freq", type=int, default=0,
                        help="Frequency of model logging to W&B (default: 0, disabled)")

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