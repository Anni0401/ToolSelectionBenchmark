import unsloth
from unsloth import FastLanguageModel

import wandb
import math
from tqdm import tqdm
import argparse
import random
import json
import numpy as np
import shutil
import gc

from typing import List
from transformers.data.data_collator import DataCollatorMixin
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict

import torch.nn.functional as F

from datasets import (
    Dataset
)

from torch.nn import KLDivLoss

import os
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    get_scheduler,
    set_seed
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_from_disk


import sys
import os
import wandb
from tqdm import tqdm
from dotenv import load_dotenv

# Set up environment
load_dotenv()  # Load environment variables from .env into os.environ
app_path = os.environ.get("APP_PATH", os.path.dirname(__file__))  # Determine project root
sys.path.insert(0, os.path.abspath(app_path))  # Ensure local src modules can be imported

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()  # Disable default progress bars in datasets to declutter logs

from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    set_seed
)
import torch
from sentence_transformers import SentenceTransformer, models

import logging
import sys

# Prevent tokenizers from using parallelism (avoid hangs)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a formatter
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Check if the root logger already has handlers
if not root_logger.handlers:
    # Create a stream handler if no handlers exist
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
else:
    # Set the formatter for existing handlers
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

# Create a logger for the current module
logger = logging.getLogger(__name__)


from src.port.prompt_templates import (
    pseudo_name_mapping, 
    pseudo_name_instr_mapping, 
    PROMPT_TEMPLATES
)

from src.port.utils import (
    compute_similarity,
    compute_embeddings,
    get_gradient_norm,
    encode_query,
    embed_corpus,
    get_ndcg_scores,
    get_ndcg_scores_multi
)

from src.port.dataset_helper import (
    DatasetDownloader,
    get_eval_dataloader,
    get_train_dataloader
)

from src.port.loss_functions import (
    compute_perplexity,
    get_batch_logps,
    odds_ratio_loss,
    compute_Pr,
    get_perplexity,
    compute_loss
)

from src.port.retrieval_evaluator import DeviceAwareInformationRetrievalEvaluator

query_id_dict = dict()
apis_multi = set()


def log_to_wandb(score_dict: dict,
                 epoch: int,
                 steps: int,
                 prefix: str = "eval") -> None:
    """
    Log metrics to Weights & Biases.

    Flattens nested dictionaries for better visualization in W&B.
    Adds a prefix to metric keys (e.g., 'eval/accuracy@k/1').
    Includes epoch and step information in the log data.

    Args:
        score_dict (dict): Dictionary containing metrics to log. Can be nested.
        epoch (int): Current training epoch number.
        steps (int): Current global training step count.
        prefix (str, optional): Prefix to add to metric keys (e.g., 'eval', 'train_eval').
                                Defaults to "eval".
    """
    # ******************** W&B Run Check ********************
    if wandb.run is None:
        logger.warning("wandb.init() has not been called. Skipping wandb.log().")
        return

    log_prefix = f"{prefix}/" if prefix else ""
    logger.info(f"Logging {prefix} metrics to W&B at epoch {epoch}, step {steps}")

    # ******************** Flatten Score Dictionary ********************
    flat_scores = {}

    def flatten_dict(d, parent_key='', sep='/'):
        """Flattens a nested dictionary for W&B logging."""
        items = []
        for k, v in d.items():
            sanitized_k = str(k).replace('@', '_at_') # Sanitize keys for W&B
            new_key = f"{parent_key}{sep}{sanitized_k}" if parent_key else sanitized_k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, np.number)): # Log only numeric types
                items.append((new_key, float(v)))
        return dict(items)

    if isinstance(score_dict, dict):
        flat_scores = flatten_dict(score_dict)
    elif isinstance(score_dict, (int, float, np.number)):
        flat_scores = {"score": float(score_dict)} # Handle single score case
    else:
        logger.warning(f"Unsupported score_dict type for logging: {type(score_dict)}")
        return

    # ******************** Prepare Log Data ********************
    wandb_log_data = {f"{log_prefix}{k}": v for k, v in flat_scores.items()}
    wandb_log_data[f"{prefix}/epoch"] = epoch
    wandb_log_data[f"{prefix}/step"] = steps

    # ******************** Log Key Metrics to Console ********************
    key_metrics = {}
    # Prioritize specific important metrics for console logging
    for key, value in wandb_log_data.items():
        if any(important in key for important in ['accuracy_at_1', 'ndcg_at_1', 'loss/total', 'epoch', 'step']):
            key_metrics[key] = value
            logger.info(f"  {key}: {value}")
    
    # If few key metrics found, log a few more numeric ones for visibility
    if key_metrics and len(key_metrics) < 5:
        for key, value in wandb_log_data.items():
            if key not in key_metrics and isinstance(value, (int, float)):
                if len(key_metrics) < 10:
                    key_metrics[key] = value
                    logger.info(f"  {key}: {value}")
    
    # ******************** Log to W&B ********************
    wandb.log(wandb_log_data, step=steps)


def create_api_evaluator(triplets: list,
                         eval_api_corpus: list,
                         batch_size: int = 16,
                         name: str = 'eval',
                         k_values_accuracy: list = [1, 3, 5, 10],
                         k_values_ndcg: list = [1, 3, 5, 10]) -> DeviceAwareInformationRetrievalEvaluator:
    """
    Construct an Information Retrieval (IR) evaluator using the sentence-transformers format.

    Maps API descriptions to document IDs and builds the necessary query/corpus/relevant_docs
    structures required by the DeviceAwareInformationRetrievalEvaluator.

    Args:
        triplets (list): A list of tuples, where each tuple is (query, positive_api, negative_api).
        eval_api_corpus (list): A list of unique API descriptions (strings) forming the corpus.
        batch_size (int, optional): Batch size for embedding computation during evaluation. Defaults to 16.
        name (str, optional): Name for the evaluator (used in logging). Defaults to 'eval'.
        k_values_accuracy (list, optional): List of integers 'k' for accuracy@k calculation. Defaults to [1, 3, 5, 10].
        k_values_ndcg (list, optional): List of integers 'k' for NDCG@k calculation. Defaults to [1, 3, 5, 10].

    Returns:
        DeviceAwareInformationRetrievalEvaluator: An initialized evaluator instance ready for use.

    Raises:
        ValueError: If no queries or no corpus documents can be generated from the inputs.
    """
    # ******************** Initialize Structures ********************
    queries = {}
    corpus = {}
    relevant_docs = {}

    # ******************** Build Corpus Mapping ********************
    # Map each unique API description string to a unique document ID (e.g., "doc_0", "doc_1")
    api_to_doc_id = {api_desc: f"doc_{i}" for i, api_desc in enumerate(eval_api_corpus)}
    # Create the corpus dictionary {doc_id: api_description}
    corpus = {doc_id: api_desc for api_desc, doc_id in api_to_doc_id.items()}

    # ******************** Build Query and Relevant Docs Mappings ********************
    query_count = 0
    anchor_to_query_id = {} # Map query strings to unique query IDs (e.g., "q_0", "q_1")

    logger.info(f"Creating evaluator '{name}' with {len(triplets)} triplets / {len(corpus)} corpus items.")

    warning_count = 0
    for anchor, positive, _ in triplets: # Iterate through (query, positive_api, negative_api)
        # Assign a unique query ID if this query hasn't been seen before
        if anchor not in anchor_to_query_id:
            query_id = f"q_{query_count}"
            anchor_to_query_id[anchor] = query_id
            queries[query_id] = anchor # Store {query_id: query_string}
            relevant_docs[query_id] = set() # Initialize relevant docs set for this query
            query_count += 1
        else:
            query_id = anchor_to_query_id[anchor]

        # Map the positive API description to its doc_id and add it to the relevant set for the query
        if positive in api_to_doc_id:
            pos_doc_id = api_to_doc_id[positive]
            relevant_docs[query_id].add(pos_doc_id)
        else:
            # Warn if a positive API from triplets is not found in the provided corpus
            warning_count += 1
            if warning_count <= 3:
                logger.warning(f"API not in corpus: '{positive[:30]}...'")
            elif warning_count == 4:
                logger.warning(f"Additional missing APIs found. Suppressing further warnings.")

    # ******************** Validation Checks ********************
    if not queries:
        logger.error("No queries generated for the evaluator. Check input triplets and corpus.")
        raise ValueError("Cannot create evaluator with no queries.")
    if not corpus:
        logger.error("Corpus is empty. Cannot create evaluator.")
        raise ValueError("Cannot create evaluator with empty corpus.")
    if not relevant_docs:
        logger.warning("Relevant docs mapping is empty. Evaluator might not produce meaningful results.")

    # ******************** Initialize Evaluator ********************
    evaluator = DeviceAwareInformationRetrievalEvaluator(
        queries,
        corpus,
        relevant_docs,
        batch_size=batch_size,
        name=name,
        show_progress_bar=(len(queries) > 10), # Show progress bar only for larger evaluations
        ndcg_at_k=k_values_ndcg,
        accuracy_at_k=k_values_accuracy
    )

    logger.info(f"Created evaluator '{name}' with {len(queries)} queries, {len(corpus)} corpus docs.")
    return evaluator

def run_evaluation(
    retr_model: AutoModel,
    retr_tokenizer: AutoTokenizer,
    dataset: Dataset,
    eval_api_corpus: list,
    retriever_max_seq_length: int,
    eval_batch_size: int,
    preprocessing_batch_size: int, # Note: This param seems unused here, consider removing if not needed
    device: str,
    k_eval_values_accuracy: list = [1, 3, 5],
    k_eval_values_ndcg: list = [1, 3, 5],
    dataset_name: str = "toole", # Note: This param seems unused here, consider removing if not needed
    eval_name: str = "eval",
    epoch: int = 0,
    steps: int = 0
):
    """
    Perform retrieval evaluation on a given dataset split using the provided model.

    1. Selects the appropriate dataset split ('train' for train_eval, 'test' or 'validation' otherwise).
    2. Generates evaluation triplets (query, positive_api, random_negative_api) ensuring the negative
       is not a known positive for that query.
    3. Wraps the Hugging Face retrieval model into a SentenceTransformer instance for compatibility
       with the evaluator (saves/reloads temporarily).
    4. Uses `create_api_evaluator` to set up the evaluation framework.
    5. Computes retrieval metrics (Accuracy@k, NDCG@k) using the evaluator.
    6. Logs the scores to W&B via `log_to_wandb`.
    7. Returns the computed ranks and NDCG scores.

    Args:
        retr_model (AutoModel): The retrieval model (e.g., RoBERTa) to evaluate.
        retr_tokenizer (AutoTokenizer): The tokenizer corresponding to the retrieval model.
        dataset (Dataset): The Hugging Face dataset containing 'query' and 'api_description' columns.
        eval_api_corpus (list): List of unique API descriptions in the evaluation corpus.
        retriever_max_seq_length (int): Max sequence length for the retriever tokenizer (unused directly here, but implicitly by the model).
        eval_batch_size (int): Batch size for embedding computation during evaluation.
        preprocessing_batch_size (int): Batch size used during data preprocessing (unused in this function).
        device (str): The device to run evaluation on ('cuda' or 'cpu').
        k_eval_values_accuracy (list, optional): List of k values for Accuracy@k. Defaults to [1, 3, 5].
        k_eval_values_ndcg (list, optional): List of k values for NDCG@k. Defaults to [1, 3, 5].
        dataset_name (str, optional): Name of the dataset (unused in this function). Defaults to "toole".
        eval_name (str, optional): Name for this evaluation run (e.g., 'eval', 'train_eval'). Defaults to "eval".
        epoch (int, optional): Current epoch number (for logging). Defaults to 0.
        steps (int, optional): Current global step number (for logging). Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - list: Accuracy@k scores (as percentages).
            - list: NDCG@k scores.
            - dict: The raw scores dictionary returned by the evaluator.
            Returns ([], [], {}) if no triplets could be generated.
    """
    logger.info(f"Running evaluation: {eval_name}")
    retr_model.eval() # Ensure the model is in evaluation mode
    # ******************** Select Dataset Split ********************
    
    if eval_name == "train_eval":
        eval_split = dataset["train"]
    else: # For "eval" or "initial_eval"
        # Prioritize 'validation' split for queries, then 'test', to match train_mnrl.py's dev_split_name logic
        if "validation" in dataset:
            eval_split = dataset["validation"]
            logger.info(f"Using 'validation' split for queries in '{eval_name}' evaluation.")
        elif "test" in dataset:
            eval_split = dataset["test"]
            logger.info(f"Using 'test' split for queries in '{eval_name}' evaluation ('validation' not found).")
        else:
            # This should ideally not happen if dataset structure is validated earlier
            logger.error(f"Neither 'validation' nor 'test' split found for '{eval_name}' evaluation. Cannot proceed.")
            return [], [], {}


    triplets = []
    corpus_set = set(eval_api_corpus) # Use a set for efficient lookup

    print(f"Evaluation split size: {len(eval_split)}")
    print(f"Corpus size: {len(corpus_set)}")
    
    # ******************** Generate Evaluation Triplets ********************
    # Step 1: Build a mapping from each query to all its known positive APIs in the corpus
    query_to_positives = {}
    for row in eval_split:
        query = row.get("query_for_retrieval")
        positive_api = row.get("api_description")
        # Ensure query and positive_api exist and the positive_api is in the evaluation corpus
        if not query or not positive_api or positive_api not in corpus_set:
            continue
        if query not in query_to_positives:
            query_to_positives[query] = set()
        query_to_positives[query].add(positive_api)
    
    logger.info(f"Found {len(query_to_positives)} unique queries with positive examples in the corpus for split '{eval_name}'")

    # Step 2: Create triplets (query, positive, negative)
    for row in eval_split:
        query = row.get("query_for_retrieval")
        positive_api = row.get("api_description")
        # Skip if data is invalid or positive not in corpus
        if not query or not positive_api or positive_api not in corpus_set:
            continue
        
        triplets.append((query, positive_api, "")) # Negative is empty for evaluation triplets

    if not triplets:
        logger.warning(f"No triplets available for evaluation ({eval_name}). Skipping evaluation.")
        return [], [], {}
    
    # Print a few example triplets
    logger.info(f"Generated {len(triplets)} triplets for evaluation '{eval_name}'. Examples:")
    for i, (q, p, n) in enumerate(triplets[:3]): # Print the first 3 triplets
        logger.info(f"  Triplet {i+1}:")
        logger.info(f"    Query: {q[:100]}...") # Print first 100 chars
        logger.info(f"    Positive: {p[:100]}...") # Print first 100 chars
        # logger.info(f"    Negative: {n[:100]}...") # Negative is empty here

    # ******************** Prepare SentenceTransformer Wrapper ********************
    logger.info(f"Preparing SentenceTransformer wrapper for evaluation")
    # Reuse the already-loaded (quantized) retr_model in-place instead of saving/reloading a
    # second full-precision copy from disk (that previously added ~16GB of extra GPU memory
    # per evaluation call and was never reclaimed in time, causing OOMs during training).
    # merge_adapter() merges the LoRA delta into the base weights in-place (reversible via
    # unmerge_adapter()), so we can wrap the SAME live model/tensors for embedding without
    # ever creating a second copy.
    if isinstance(retr_model, PeftModel):
        retr_model.merge_adapter()

    transformer = models.Transformer.__new__(models.Transformer)
    torch.nn.Module.__init__(transformer)
    transformer.config_keys = ["max_seq_length", "do_lower_case"]
    transformer.do_lower_case = False
    transformer.backend = "torch"
    transformer.auto_model = retr_model
    transformer.tokenizer = retr_tokenizer
    transformer.max_seq_length = retriever_max_seq_length

    # Define pooling strategy (last-token pooling, required for decoder-style embedding models like Qwen)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_lasttoken=True,
        pooling_mode_cls_token=False,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
        pooling_mode_mean_sqrt_len_tokens=False
    )
    normalize = models.Normalize() # Add L2 normalization layer
    st_retr_model = SentenceTransformer(modules=[transformer, pooling, normalize], device=device)

    # ******************** Create Evaluator Instance ********************
    logger.info(f"Creating evaluator instance for '{eval_name}' using the generated triplets.") # Added log before evaluator creation
    evaluator = create_api_evaluator(
        triplets,
        eval_api_corpus,
        batch_size=eval_batch_size,
        name=eval_name,
        k_values_accuracy=k_eval_values_accuracy,
        k_values_ndcg=k_eval_values_ndcg
    )

    # ******************** Compute Corpus Embeddings ********************
    logger.info(f"Computing corpus embeddings for '{eval_name}'")
    corpus_embeddings = embed_corpus(
        retr_model=st_retr_model,
        retr_tokenizer=retr_tokenizer,
        corpus=eval_api_corpus,
        max_length=retriever_max_seq_length,
        batch_size=eval_batch_size,
        device=device
    )

    # ******************** Compute and Log Scores ********************
    logger.info(f"Computing evaluation scores for '{eval_name}'")
    scores = evaluator(st_retr_model, corpus_embeddings=corpus_embeddings) # Run the evaluation
    
    # Log score structure once for clarity
    if hasattr(run_evaluation, 'first_run') == False:
        logger.info(f"Score structure example: {list(scores.keys())}")
        run_evaluation.first_run = True

    log_to_wandb(scores, epoch=epoch, steps=steps, prefix=eval_name) # Log to W&B

    # ******************** Extract Key Metrics ********************
     # pull out the flat keys directly:
    ranks = [
        scores.get(f"cosine_accuracy@{k}", 0.0) * 100
        for k in k_eval_values_accuracy
    ]
    ndcg_scores_list = [
        scores.get(f"cosine_ndcg@{k}", 0.0)
        for k in k_eval_values_ndcg
    ]

    # ******************** Cleanup ********************
    if isinstance(retr_model, PeftModel):
        retr_model.unmerge_adapter() # Undo the in-place merge to resume training with the LoRA delta active
    retr_model.train() # Set model back to train mode if it was changed by evaluator

    # st_retr_model wraps the SAME live retr_model tensors (no second copy was made), so we
    # only need to drop the wrapper objects and free the activation memory from embedding.
    del st_retr_model, transformer, pooling, normalize, evaluator, corpus_embeddings
    gc.collect()
    torch.cuda.empty_cache() # Clear GPU cache

    return ranks, ndcg_scores_list, scores

run_evaluation.first_run = False


def train(dataset: Dataset,
           dataset_name: str,
           retr_tokenizer: AutoTokenizer,
           retr_model: AutoModel,
           infer_tokenizer: AutoTokenizer,
           infer_model: AutoModelForCausalLM,
           train_api_corpus: List[str],
           eval_api_corpus: List[str],
           data_collator_completion: DataCollatorMixin,
           eval_strategy: str = "epoch",
           eval_steps: float = None,
           n_reembedding_steps: int = None,
           embedding_update_steps: int = None,
           prompt_template: str = "",
           instruction_prompt: str = "", # Note: This param seems unused here, consider removing if not needed
           lambda_loss: float = 0.2,
           beta: float = 1,
           gamma: float = 1,
           preference_weight: float = 0.1,
           num_epochs: int = 10,
           learning_rate: float = 1e-4,
           scheduler_type: str = "cosine",
           warmup_ratio: float = 0.1,
           retriever_max_seq_length: int = 514,
           inference_max_seq_length: int = 1024,
           number_of_neg_examples: int = 3,
           train_batch_size: int = 2,
           eval_batch_size: int = 2,
           preprocessing_batch_size: int = 4,
           log_freq: int = 100,
           k_eval_values_accuracy: list = [1, 3, 5],
           k_eval_values_ndcg: list = [1, 3, 5],
           device: str = "cuda",
           wandb_project_name: str = "",
           wandb_run_name: str = "",
           save_strategy: str = "epoch",
           save_steps: int = None,
           save_dir: str = "./checkpoints",
           max_checkpoints: int = None,
           weight_decay: float = 0.01,
           save_checkpoints: bool = False,
           infer_model_name: str = "gpt2",
           retr_model_name: str = "roberta-base"):
    """
    Main training loop for the PORT model.

    Handles initialization, epoch iteration, batch processing, loss calculation,
    optimization, evaluation, logging, and checkpointing.

    Args:
        dataset (Dataset): The preprocessed Hugging Face dataset.
        dataset_name (str): Name of the dataset being used.
        retr_tokenizer (AutoTokenizer): Tokenizer for the retrieval model.
        retr_model (AutoModel): The retrieval model to be trained.
        infer_tokenizer (AutoTokenizer): Tokenizer for the inference model.
        infer_model (AutoModelForCausalLM): The frozen inference model (LLM).
        train_api_corpus (List[str]): List of unique API descriptions in the training set corpus.
        eval_api_corpus (List[str]): List of unique API descriptions in the evaluation set corpus.
        data_collator_completion (DataCollatorMixin): Collator for preparing inference model inputs.
        eval_strategy (str, optional): When to run evaluation ('epoch' or 'steps'). Defaults to "epoch".
        eval_steps (float, optional): If eval_strategy='steps', fraction of steps per split/epoch
                                      after which to evaluate (0.0 < steps <= 1.0). Defaults to None.
        n_reembedding_steps (int, optional): Number of sub-splits of the training data. If set,
                                             corpus embeddings are recomputed for each split. Defaults to None (no re-splitting).
        embedding_update_steps (int, optional): Number of steps after which to update corpus embeddings. If specified, overrides n_reembedding_steps. Defaults to None.
        prompt_template (str, optional): The template string used to format prompts for the inference model. Defaults to "".
        instruction_prompt (str, optional): An instruction string prepended to prompts (unused). Defaults to "".
        lambda_loss (float, optional): Weighting factor for the preference loss term. Defaults to 0.2.
        beta (float, optional): Temperature parameter for the Q distribution softmax. Defaults to 1.
        gamma (float, optional): Temperature parameter for the Pr_retr distribution softmax. Defaults to 1.
        preference_weight (float, optional): Beta parameter for the odds ratio preference loss. Defaults to 0.1.
        num_epochs (int, optional): Total number of training epochs. Defaults to 10.
        learning_rate (float, optional): Peak learning rate for the optimizer. Defaults to 1e-4.
        scheduler_type (str, optional): Type of learning rate scheduler. Defaults to "cosine".
        warmup_ratio (float, optional): Fraction of total steps for linear warmup. Defaults to 0.1.
        retriever_max_seq_length (int, optional): Max sequence length for the retriever. Defaults to 514.
        inference_max_seq_length (int, optional): Max sequence length for the inference model. Defaults to 1024.
        number_of_neg_examples (int, optional): Number of negative examples per positive example. Defaults to 3.
        train_batch_size (int, optional): Batch size for training. Defaults to 2.
        eval_batch_size (int, optional): Batch size for evaluation. Defaults to 2.
        preprocessing_batch_size (int, optional): Batch size used during data preprocessing. Defaults to 4.
        log_freq (int, optional): Log training metrics every `log_freq` steps. Defaults to 100.
        k_eval_values_accuracy (list, optional): k values for Accuracy@k evaluation. Defaults to [1, 3, 5].
        k_eval_values_ndcg (list, optional): k values for NDCG@k evaluation. Defaults to [1, 3, 5].
        device (str, optional): Device to run training on ('cuda' or 'cpu'). Defaults to "cuda".
        wandb_project_name (str, optional): Project name for W&B logging. Defaults to "".
        wandb_run_name (str, optional): Run name for W&B logging (defaults to generated name if empty). Defaults to "".
        save_strategy (str, optional): When to save checkpoints ('epoch' or 'steps'). Defaults to "epoch".
        save_steps (int, optional): If save_strategy='steps', save every `save_steps` steps. Defaults to None.
        save_dir (str, optional): Directory to save model checkpoints. Defaults to "./checkpoints".
        max_checkpoints (int, optional): Maximum number of checkpoints to keep. Older/worse ones might be deleted. Defaults to None (keep all).
        weight_decay (float, optional): Weight decay for the AdamW optimizer. Defaults to 0.01.
        save_checkpoints (bool, optional): Whether to save model checkpoints during training. Defaults to False.
    """
    # ******************** W&B Initialization & Config Logging ********************
    # Ensure only n_reembedding_steps is used if both are defined
    if n_reembedding_steps is not None:
        actual_embedding_update_steps = None
    else:
        actual_embedding_update_steps = embedding_update_steps
    
    config = {
        "beta" : beta,
        "gamma" : gamma,
        "lr" : learning_rate,
        "weight_decay": weight_decay, # Add weight decay to config
        "scheduler_type" : scheduler_type,
        "train_batch_size" : train_batch_size,
        "eval_batch_size" : eval_batch_size,
        "lambda_loss_factor" : lambda_loss,
        "retriever_max_seq_length" : retriever_max_seq_length,
        "inference_max_seq_length" : inference_max_seq_length,
        "epochs" : num_epochs,
        "dataset" : dataset_name,
        "n_reembedding_steps": n_reembedding_steps,
        "embedding_update_steps": actual_embedding_update_steps,
        "train_data_samples" : len(dataset["train"]),
        "preference_weight": preference_weight,
        "num_neg_examples": number_of_neg_examples,
        "warmup_ratio": warmup_ratio,
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "save_checkpoints": save_checkpoints,
        "inference_model_name": infer_model_name,
        "retr_model_name": retr_model_name
    }

    # Include the swept hyperparameters in the auto-generated name so grid/sweep runs
    # (which all share the same dataset/model) remain distinguishable in the W&B UI.
    run_name = wandb_run_name or (
        f"PORTS-{dataset_name}-{infer_model_name}-{retr_model_name}"
        f"-lr{learning_rate}-ep{num_epochs}-lam{lambda_loss}-b{beta}-g{gamma}"
    )
    logger.info(f"Initializing W&B with project '{wandb_project_name}', run name: {run_name}")
    wandb.init(project=wandb_project_name, name=run_name)
    wandb.config.update(config)
    wandb.watch(retr_model, log_freq=log_freq) # Watch model gradients

    # ******************** Initial Evaluation ********************
    infer_model.eval() # Ensure inference model is in eval mode
    
    logger.info(f"Starting initial evaluations (before training)")
    # --- Evaluation on Test/Validation Set ---
    eval_config = {
        "retr_model" : retr_model,
        "retr_tokenizer" : retr_tokenizer,
        "dataset" : dataset,
        "eval_api_corpus" : eval_api_corpus,
        "retriever_max_seq_length" : retriever_max_seq_length,
        "eval_batch_size" : eval_batch_size,
        "preprocessing_batch_size" : preprocessing_batch_size,
        "device" : device,
        "k_eval_values_accuracy" : k_eval_values_accuracy,
        "k_eval_values_ndcg" : k_eval_values_ndcg,
        "dataset_name" : dataset_name,
        "eval_name": "eval",
        "epoch": 0,
        "steps": 0
    }
    ranks_eval, ndcg_eval, scores_eval = run_evaluation(**eval_config)

    logger.info("--- Initial Evaluation Summary (eval split) ---")
    logger.info(f"  Accuracy@k values ({k_eval_values_accuracy}): {[f'{x:.2f}%' for x in ranks_eval]}")
    logger.info(f"  NDCG@k values ({k_eval_values_ndcg}): {[f'{x:.4f}' for x in ndcg_eval]}")
    if wandb.run: # Log to W&B if initialized
        log_to_wandb(scores_eval, epoch=0, steps=0, prefix="initial_eval")

    # --- Evaluation on Training Set ---
    logger.info(f"Starting Initial Evaluation (Train)")
    train_eval_config = eval_config.copy()
    train_eval_config["eval_api_corpus"] = train_api_corpus
    train_eval_config["eval_name"] = "train_eval"
    ranks_train_eval, ndcg_train_eval, scores_train_eval = run_evaluation(**train_eval_config)

    logger.info("--- Initial Evaluation Summary (train_eval split) ---")
    logger.info(f"  Accuracy@k values ({k_eval_values_accuracy}): {[f'{x:.2f}%' for x in ranks_train_eval]}")
    logger.info(f"  NDCG@k values ({k_eval_values_ndcg}): {[f'{x:.4f}' for x in ndcg_train_eval]}")
    if wandb.run: # Log to W&B if initialized
        log_to_wandb(scores_train_eval, epoch=0, steps=0, prefix="initial_train_eval")

    # ******************** Optimizer and Scheduler Setup ********************
    # Only optimize trainable params (e.g. LoRA adapters when the base model is frozen)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, retr_model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay # Use the parameter here
    )

    ds_length = len(dataset["train"])
    # Calculate total training steps
    n_iters = math.ceil(ds_length / train_batch_size) # Use math.ceil for accurate step count
    num_training_steps = num_epochs * n_iters
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    logger.info(f"Dataset size: {ds_length}, Steps per epoch: {n_iters}")
    logger.info(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps} ({warmup_ratio*100:.1f}%)")

    lr_scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ******************** Loss Function Setup ********************
    kl_div = KLDivLoss(reduction='none') # Use 'none' reduction for manual aggregation

    # ******************** Checkpoint Management Setup ********************
    saved_checkpoints = [] # List to track saved checkpoints for potential pruning
    os.makedirs(save_dir, exist_ok=True) # Ensure save directory exists

    # ******************** Epoch Loop ********************
    global_step_counter = 0 # Use a single counter across epochs and splits
    for epoch in range(num_epochs):
        retr_model.train() # Set model to training mode
        data_splits = []

        # ******************** Data Splitting (for Re-embedding) ********************
        # Prioritize n_reembedding_steps if set
        if n_reembedding_steps and n_reembedding_steps > 0:
            # Use n_reembedding_steps for splitting
            subsplit_len = ds_length // n_reembedding_steps
            data_subsplits_lens = [subsplit_len] * n_reembedding_steps
            # Add the remainder to the last split
            remainder = ds_length % n_reembedding_steps
            if remainder > 0:
                 data_subsplits_lens[-1] += remainder # Add remainder to the last split's length

            logger.info(f"Creating {len(data_subsplits_lens)} data sub-splits for Epoch {epoch+1} using n_reembedding_steps={n_reembedding_steps} (lengths: {data_subsplits_lens})")
            current_index = 0
            for idx, _len in enumerate(data_subsplits_lens):
                _start = current_index
                _end = current_index + _len
                _ds_sample = dataset["train"].select(range(_start, _end))
                data_splits.append(DatasetDict({"train": _ds_sample}))
                current_index = _end
        else:
            # No re-embedding, use the whole dataset for the epoch
            data_splits = [dataset]
            logger.info(f"Using full dataset for Epoch {epoch+1} (no n_reembedding_steps specified)")

        logger.info(f"Starting training epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        epoch_steps = 0
        
        # ******************** Calculate Total Steps per Epoch ********************
        total_epoch_steps = 0
        for ds_split in data_splits:
            train_data_config = {
                "dataset": ds_split,
                "api_corpus_list": train_api_corpus,
                "retr_model": retr_model,
                "retrieval_max_length": retriever_max_seq_length,
                "generateor_max_length": inference_max_seq_length,
                "retrieval_tokenizer": retr_tokenizer,
                "inference_tokenizer": infer_tokenizer,
                "epoch_number": epoch,
                "batch_size": train_batch_size,
                "prompt_template": prompt_template,
                "num_neg_examples": number_of_neg_examples,
                "preprocessing_batch_size": preprocessing_batch_size
            }
            # Get dataloader length to calculate steps
            total_epoch_steps += len(get_train_dataloader(**train_data_config))
        
        # ******************** Step-based Evaluation Setup ********************
        evaluation_step_interval = None
        if eval_strategy == "steps" and eval_steps is not None:
            if not (0 < eval_steps <= 1):
                # Ensure eval_steps is a valid fraction
                raise ValueError("eval_steps must be a float between 0 (exclusive) and 1 (inclusive) when eval_strategy is 'steps'")
            # Calculate the step interval for evaluation across the entire epoch
            evaluation_step_interval = max(1, int(total_epoch_steps * eval_steps))
            logger.info(f"Evaluation strategy: 'steps'. Evaluating every {evaluation_step_interval} steps ({eval_steps*100:.2f}% of {total_epoch_steps} total steps in epoch).")
        
        # Reset step counter for each epoch (for epoch-based evaluation)
        epoch_step_counter = 0

        # ******************** Data Split Loop ********************
        for split_idx, ds_split in enumerate(data_splits):
            logger.info(f">> Processing data split {split_idx+1}/{len(data_splits)}")
            
            train_data_config = {
                "dataset": ds_split,
                "api_corpus_list": train_api_corpus,
                "retr_model": retr_model, # Pass the original HF model to dataloader if needed elsewhere
                "retrieval_max_length": retriever_max_seq_length,
                "generateor_max_length": inference_max_seq_length,
                "retrieval_tokenizer": retr_tokenizer,
                "inference_tokenizer": infer_tokenizer,
                "epoch_number": epoch,
                "batch_size": train_batch_size,
                "prompt_template": prompt_template,
                "num_neg_examples": number_of_neg_examples,
                "preprocessing_batch_size": preprocessing_batch_size
            }
            triplet_dataloader = get_train_dataloader(**train_data_config)

            steps_per_split = len(triplet_dataloader)
            epoch_steps += steps_per_split # Accumulate steps for epoch avg loss
            
            # ******************** Batch Loop (Training Step) ********************
            pbar = tqdm(enumerate(triplet_dataloader), 
                        total=steps_per_split, 
                        desc=f"Epoch {epoch+1}/{num_epochs} (Split {split_idx+1})",
                        miniters=max(1, steps_per_split // 20)) # Update progress bar less frequently for speed

            for batch_idx, batch in pbar:
                global_step_counter += 1 # Increment global step counter
                epoch_step_counter += 1 # Increment epoch step counter

                # --- Move batch tensors to device ---
                queries = {k: v.to(device) for k, v in batch["query"].items()}
                pos_docs = {k: v.to(device) for k, v in batch["positive"].items()}

                # Handle negatives via HF DataCollatorWithPadding
                if isinstance(batch["negative"], list):
                    neg_docs_processed = DataCollatorWithPadding(tokenizer=retr_tokenizer)(
                        batch["negative"]
                    )
                    neg_docs_processed = {k: v.to(device) for k, v in neg_docs_processed.items()}
                else:
                    neg_docs_processed = {k: v.to(device) for k, v in batch["negative"].items()}

                bs = queries["input_ids"].size(0)

                # --- Compute Similarities ---
                pos_similarity = compute_similarity(retr_model, queries, pos_docs).view(bs, -1) # [bs, 1]

                neg_similarity_list = []
                n_neg_docs = neg_docs_processed["input_ids"].shape[1] # Number of negatives per query

                for nid in range(n_neg_docs):
                    # Select data for the n-th negative example across the batch
                    current_neg_data = {
                        k: neg_docs_processed[k][:, nid, :] for k in ['input_ids', 'attention_mask']
                    }
                    this_neg_similarity = compute_similarity(retr_model, queries, current_neg_data).view(bs, -1) # [bs, 1]
                    neg_similarity_list.append(this_neg_similarity)
                
                neg_similarity = torch.stack(neg_similarity_list, dim=-1) # Shape: [bs, 1, num_neg]
                neg_similarity = neg_similarity.squeeze(1) # Shape: [bs, num_neg]

                # Concatenate positive and negative similarities: [bs, 1 + num_neg]
                similarities = torch.cat((pos_similarity, neg_similarity), dim=-1)

                # Apply temperature scaling (gamma)
                similarities = similarities / gamma

                # --- Compute Retrieval Probability (Pr_retr) ---
                Pr_retr = compute_Pr(
                    similarities = similarities,
                    axis = -1 # Softmax over the last dimension (pos + neg docs)
                ) # Shape: [bs, 1 + num_neg]

                # --- Prepare Inference Model Inputs ---
                input_prompt_pos = batch["q_pos_prompt"]
                input_prompt_neg = batch["q_neg_prompt"]
                input_prompt_pos = {k : input_prompt_pos[k].to(device) for k in input_prompt_pos}
                input_prompt_neg = [{k : neg_docs_trip[k].to(device) for k in neg_docs_trip} for neg_docs_trip in input_prompt_neg]
                
                # --- Compute Perplexities (Q) using Frozen LLM ---
                pos_perplexity = []
                neg_perplexity_list = []

                with torch.no_grad(): # Ensure no gradients are computed for the inference model
                    # --- Positive Perplexity ---
                    # Create a temporary dataset/dataloader for the positive prompts for the collator
                    pos_dataloader = DataLoader(
                        Dataset.from_dict(input_prompt_pos), 
                        shuffle=False, 
                        batch_size=bs, # Process the whole batch at once
                        collate_fn=data_collator_completion) # Use the completion-only collator
                    
                    pos_data_batch = next(iter(pos_dataloader)) # Get the collated batch
                    pos_data_batch = {k: v.to(device) for k, v in pos_data_batch.items()}
                    labels = pos_data_batch.pop("labels") # Labels prepared by collator

                    outputs_pos = infer_model(**pos_data_batch)

                    pos_perplexity = get_perplexity(outputs=outputs_pos, 
                                                    input_ids=labels,
                                                    attention_mask=pos_data_batch["attention_mask"],
                                                    padding_token_ids=retr_tokenizer.pad_token_id)

                    del outputs_pos, pos_data_batch, labels, pos_dataloader
                    torch.cuda.empty_cache()

                    # --- Negative Perplexity ---
                    for n_id in range(n_neg_docs):
                        neg_data = next(iter(DataLoader(
                            Dataset.from_dict(
                                {k: torch.stack([input_prompt_neg[bid][k][n_id,:] for bid in range(bs)]) 
                                for k in ["input_ids", "attention_mask"]}), 
                            shuffle=False, 
                            batch_size=bs, 
                            collate_fn=data_collator_completion)))
                        
                        neg_data = {k: v.to(device) for k, v in neg_data.items()}
                        labels = neg_data.pop("labels")
                        
                        outputs_neg = infer_model(**neg_data)

                        neg_perplexity_list.append(get_perplexity(outputs=outputs_neg, 
                                                                  input_ids=labels,
                                                                  attention_mask=neg_data["attention_mask"],
                                                                  padding_token_ids=infer_tokenizer.pad_token_id)) # Use infer tokenizer pad id

                        del outputs_neg, neg_data, labels
                        torch.cuda.empty_cache()

                # Stack negative perplexities: [bs, num_neg]
                neg_perplexity = torch.stack(neg_perplexity_list, dim=-1)
                # Concatenate positive and negative perplexities: [bs, 1 + num_neg]
                concat_perplexities = torch.cat((pos_perplexity.unsqueeze(-1), neg_perplexity), dim=-1)
                
                # Apply temperature scaling (beta)
                concat_perplexities = concat_perplexities / beta
                
                # --- Compute Q Distribution ---
                Q = F.softmax(concat_perplexities, dim=-1) # Shape: [bs, 1 + num_neg]

                del concat_perplexities, pos_perplexity, neg_perplexity, neg_perplexity_list
                torch.cuda.empty_cache()

                # --- Compute KL Divergence Loss (RePlug Component) ---
                ppl_pr_KL_loss = compute_loss(Q, Pr_retr, kl_div) # Check compute_loss implementation details

                # --- Compute Preference Loss (ORPO Component) ---
                pref_loss_total = 0
                pos_rewards_list, neg_rewards_list = [], []
                pref_ratio_total, mean_prob_ratio_total = 0, 0

                pos_retrieval_prob = Pr_retr[:, 0] # Probability of retrieving the positive doc
                neg_retrieval_probs = Pr_retr[:, 1:] # Probabilities of retrieving negative docs [bs, num_neg]

                for neg_i in range(n_neg_docs):
                    neg_retrieval_prob = neg_retrieval_probs[:, neg_i] # Probability for the i-th negative
                    
                    # Calculate odds ratio loss for this positive-negative pair
                    _pref_loss, _pos_reward, _neg_reward, _pref_ratio, _mean_prob_ratio = odds_ratio_loss(
                        positive_retr_log_prob=pos_retrieval_prob.log(), # Use log probabilities
                        negative_retr_log_prob=neg_retrieval_prob.log(),
                        beta=preference_weight # Use the specific beta for ORPO loss
                    )
                    pref_loss_total += _pref_loss.mean() # Accumulate mean loss over the batch
                    pos_rewards_list.append(_pos_reward.mean()) # Log mean rewards
                    neg_rewards_list.append(_neg_reward.mean())
                    pref_ratio_total += _pref_ratio # Log mean ratio
                    mean_prob_ratio_total += _mean_prob_ratio # Log mean prob ratio

                # Average the preference loss and stats over the number of negatives
                pref_loss = pref_loss_total / n_neg_docs
                avg_pref_ratio = pref_ratio_total / n_neg_docs
                avg_mean_prob_ratio = mean_prob_ratio_total / n_neg_docs
                
                # --- Combine Losses ---
                loss = ppl_pr_KL_loss - lambda_loss * pref_loss

                # --- Backpropagation and Optimization ---
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retr_model.parameters(), max_norm=1.0) # Gradient clipping
                optimizer.step() # Update retriever weights
                lr_scheduler.step() # Update learning rate
                
                grad_norm = get_gradient_norm(retr_model) # Get gradient norm for logging

                optimizer.zero_grad() # Clear gradients for next step
                
                # --- Log Metrics ---
                epoch_loss += loss.item() # Accumulate loss for epoch average

                pos_rewards_tensor = torch.tensor(pos_rewards_list, device=device) # Use list means
                neg_rewards_tensor = torch.tensor(neg_rewards_list, device=device)
                retrieval_accuracy = (pos_rewards_tensor > neg_rewards_tensor).float().mean() # Average over negatives

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'RePlug': f"{ppl_pr_KL_loss.item():.4f}", 
                    "ORPO": f"{-pref_loss.item():.4f}" # Log negative ORPO loss (as it's subtracted)
                })
                
                if global_step_counter % log_freq == 0:
                    log_metrics = {
                        "loss/total": loss.item(),
                        "loss/replug_kl": ppl_pr_KL_loss.item(),
                        "loss/orpo": -pref_loss.item(), # Log the negative value consistent with formula
                        "orpo/avg_pref_ratio": avg_pref_ratio,
                        "orpo/avg_retrieval_accuracy_proxy": retrieval_accuracy.cpu(), # Note: Proxy metric
                        "orpo/avg_mean_prob_ratio": avg_mean_prob_ratio,
                        "probabilities/positive_retrieval_mean": pos_retrieval_prob.mean().cpu(),
                        "probabilities/negative_retrieval_mean": neg_retrieval_probs.mean().cpu(), # Mean over all negatives
                        "Q_values/positive_mean": Q[:,0].mean().cpu(),
                        "Q_values/negative_mean": Q[:,1:].mean().cpu(), # Mean over all negatives
                        "similarity/positive_mean": pos_similarity.mean().cpu(),
                        "similarity/negative_mean": neg_similarity.mean().cpu(), # Mean over all negatives
                        "optimizer/gradient_norm": grad_norm,
                        "optimizer/learning_rate": optimizer.param_groups[0]['lr'],
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step_counter # Use global step counter
                    }
                    wandb.log(log_metrics, step=global_step_counter)

                # --- Cleanup Batch Tensors ---
                del Q, Pr_retr, ppl_pr_KL_loss, pref_loss, loss, pos_retrieval_prob, neg_retrieval_probs
                del avg_pref_ratio, retrieval_accuracy, avg_mean_prob_ratio
                del similarities, pos_similarity, neg_similarity
                del queries, pos_docs, neg_docs_processed, input_prompt_pos, input_prompt_neg
                torch.cuda.empty_cache()

                # ******************** Step-based Evaluation Trigger ********************
                if eval_strategy == "steps" and evaluation_step_interval is not None and global_step_counter % evaluation_step_interval == 0:
                    logger.info(f"--- Running Step-Based Evaluation (Step: {global_step_counter}, {((epoch_step_counter/total_epoch_steps)*100):.1f}% through epoch) ---")
                    eval_config_step = {
                        "retr_model" : retr_model,
                        "retr_tokenizer" : retr_tokenizer,
                        "dataset" : dataset, # Use the full dataset for eval consistency
                        "eval_api_corpus" : eval_api_corpus,
                        "retriever_max_seq_length" : retriever_max_seq_length,
                        "eval_batch_size" : eval_batch_size,
                        "preprocessing_batch_size" : preprocessing_batch_size,
                        "device" : device,
                        "k_eval_values_accuracy" : k_eval_values_accuracy,
                        "k_eval_values_ndcg" : k_eval_values_ndcg,
                        "dataset_name" : dataset_name,
                        "eval_name": "eval",
                        "epoch": epoch + 1,
                        "steps": global_step_counter # Log with global step
                    }
                    run_evaluation(**eval_config_step)

                    logger.info(f"--- Running Step-Based Evaluation (Train) (Step: {global_step_counter}) ---")
                    train_eval_config_step = eval_config_step.copy()
                    train_eval_config_step["eval_api_corpus"] = train_api_corpus
                    train_eval_config_step["eval_name"] = "train_eval"
                    run_evaluation(**train_eval_config_step)
                    
                    retr_model.train() # Ensure model is back in train mode after eval

                # ******************** Step-based Checkpoint Saving ********************
                if save_strategy == "steps" and save_steps and global_step_counter % save_steps == 0 and save_checkpoints:
                    save_path = os.path.join(save_dir, f"checkpoint-step-{global_step_counter}")
                    logger.info(f"Saving checkpoint at step {global_step_counter} to {save_path}")
                    retr_model.save_pretrained(save_path)

            torch.cuda.empty_cache()

            del triplet_dataloader # Free memory
            torch.cuda.empty_cache()

        # ******************** Epoch-based Checkpoint Saving ********************
        if save_strategy == "epoch" and save_checkpoints:
            ckpt_name = f"checkpoint-epoch-{epoch+1}"
            save_path = os.path.join(save_dir, ckpt_name)
            logger.info(f"Saving checkpoint at end of epoch {epoch+1} to {save_path}")
            retr_model.save_pretrained(save_path)

            saved_checkpoints.append(save_path) # Store the full path
            if max_checkpoints and len(saved_checkpoints) > max_checkpoints:
                ckpt_to_remove = saved_checkpoints.pop(0) # Remove the oldest checkpoint path
                if os.path.exists(ckpt_to_remove):
                    logger.info(f"Max checkpoints ({max_checkpoints}) reached. Removing oldest epoch checkpoint: {ckpt_to_remove}")
                    
                    try:
                        shutil.rmtree(ckpt_to_remove)
                    except OSError as e:
                        logger.error(f"Error removing checkpoint {ckpt_to_remove}: {e}")
                else:
                     logger.warning(f"Tried to remove checkpoint {ckpt_to_remove}, but it was not found.")

        # ******************** Epoch-based Evaluation Trigger ********************
        if eval_strategy == "epoch":
            logger.info(f"--- Running Epoch-End Evaluation (Epoch: {epoch+1}) ---")
            eval_config_epoch = {
                "retr_model" : retr_model,
                "retr_tokenizer" : retr_tokenizer,
                "dataset" : dataset, # Use full dataset
                "eval_api_corpus" : eval_api_corpus,
                "retriever_max_seq_length" : retriever_max_seq_length,
                "eval_batch_size" : eval_batch_size,
                "preprocessing_batch_size" : preprocessing_batch_size,
                "device" : device,
                "k_eval_values_accuracy": k_eval_values_accuracy,
                "k_eval_values_ndcg": k_eval_values_ndcg,
                "dataset_name": dataset_name,
                "eval_name": "eval",
                "epoch": epoch + 1,
                "steps": global_step_counter # Log with global step at epoch end
            }
            run_evaluation(**eval_config_epoch)

            logger.info(f"--- Running Epoch-End Evaluation (Train) (Epoch: {epoch+1}) ---")
            train_eval_config_epoch = eval_config_epoch.copy()
            train_eval_config_epoch["eval_api_corpus"] = train_api_corpus
            train_eval_config_epoch["eval_name"] = "train_eval"
            run_evaluation(**train_eval_config_epoch)
            
            retr_model.train() # Ensure model is back in train mode

        epoch_avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Avg loss: {epoch_avg_loss:.4f}")
        wandb.log({"epoch/average_loss": epoch_avg_loss, "epoch": epoch + 1}, step=global_step_counter)

    logger.info("Training and evaluations finished.")
    wandb.finish() # Ensure W&B run is closed


def main():
    """
    Main entry point for the PORT training script.

    Parses command-line arguments, sets up models, tokenizers, datasets,
    and initiates the training process by calling the `train` function.
    """
    # ******************** Argument Parsing ********************
    parser = argparse.ArgumentParser(description='PORT training script')
    # --- Dataset Args ---
    parser.add_argument('--dataset', type=str, default="bfcl", choices=["bfcl", "apibank", "apibench", "octopus", "octopus-overlap", "toole", "toole-overlap", "toolbench", "toolbench_1", "toolbench_2", "toolbench_3", "toole_90_10", "toole_85_15", "toole_75_25", "toole_70_30", "toole_50_50", "toole_35_65"], help='Dataset name for training and evaluation.')
    parser.add_argument('--max_train_samples', type=int, default=None, help="Maximum number of training instances to use (samples randomly). Default: use all.")
    parser.add_argument('--max_eval_samples', type=int, default=None, help="Maximum number of evaluation instances to use (samples randomly). Default: use all.")

    # --- Model Args ---
    parser.add_argument('--inference_model_name', type=str, default="llama3.2", choices=["llama3-8B", "codestral-22B", "gemma2-2B", "groqLlama3Tool-8B","llama3.2","gemma3","qwen3"], help="Pseudo-name of the generative model (LLM) used for perplexity calculation.")
    parser.add_argument('--retrieval_model_name', type=str, default="FacebookAI/roberta-base", help="Hugging Face model name or path for the retrieval model to be trained.")
    parser.add_argument('--retriever_max_seq_length', type=int, default=514, help="Max sequence length for the retriever model tokenizer.")
    parser.add_argument('--inference_max_seq_length', type=int, default=1024, help="Max sequence length for the inference model tokenizer.")
    parser.add_argument('--load_in_4bit', action='store_true', default=False, help="Load the inference model using 4-bit quantization (BitsAndBytes).")

    # --- LoRA / QLoRA Args (retriever) ---
    parser.add_argument('--use_lora', action='store_true', default=False, help="Wrap the retrieval model with a LoRA adapter (via peft) instead of full fine-tuning.")
    parser.add_argument('--use_qlora', action='store_true', default=False, help="Load the retrieval model in 4-bit (BitsAndBytes) before applying LoRA. Requires --use_lora.")
    parser.add_argument('--lora_r', type=int, default=16, help="LoRA rank.")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha (scaling factor).")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument('--lora_target_modules', nargs="+", type=str, default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help="Module names to attach LoRA adapters to.")

    # --- Training Control Args ---
    parser.add_argument('--do_train', action='store_true', default=False, help="Whether to run the training loop (currently always runs if script is executed).") # Consider if needed
    parser.add_argument('--do_eval', action='store_true', default=False,  help="Whether to run the evaluation loop (evaluation is integrated into training).") # Consider if needed
    parser.add_argument('--n_epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--n_reembedding_steps', type=int, default=None, help="Number of data splits per epoch for potential corpus re-embedding (if applicable in dataloader). Default: None (no splitting).")
    parser.add_argument('--embedding_update_steps', type=int, default=None, help="Number of steps after which to update corpus embeddings. If specified, overrides n_reembedding_steps. Default: None.")

    # --- Evaluation Args ---
    parser.add_argument('--eval_strategy', type=str, default="epoch", choices=["epoch", "steps"], help="Strategy for running evaluation during training ('epoch' or 'steps').")
    parser.add_argument('--eval_steps', type=float, default=None, help="If eval_strategy='steps', evaluate every X fraction of steps within a split/epoch (e.g., 0.1 for every 10%).")
    parser.add_argument("--k_eval_values_accuracy", nargs="+", type=int, default=[1, 3, 5], help="Values of k for Accuracy@k evaluation.")
    parser.add_argument("--k_eval_values_ndcg", nargs="+", type=int, default=[1, 3, 5], help="Values of k for NDCG@k evaluation.")

    # --- Checkpoint Args ---
    parser.add_argument('--save_strategy', type=str, default="epoch", choices=["epoch", "steps"], help="Strategy for saving model checkpoints ('epoch' or 'steps').")
    parser.add_argument('--save_steps', type=int, default=None, help="If save_strategy='steps', save a checkpoint every N steps.")
    parser.add_argument('--save_dir', type=str, default="./checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument('--max_checkpoints', type=int, default=None, help="Maximum number of checkpoints to keep (based on strategy, e.g., removes oldest). Default: keep all.")
    parser.add_argument('--save_checkpoints', action='store_true', default=False, help="Whether to save model checkpoints during training.")

    # --- Optimizer & Scheduler Args ---
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the AdamW optimizer.")
    parser.add_argument('--lr_type', type=str, default="cosine", help="Learning rate scheduler type (e.g., 'linear', 'cosine').")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Fraction of total training steps used for linear learning rate warmup (0.0 to 1.0).")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for the AdamW optimizer.")

    # --- Batch Size & Tokenizer Args ---
    parser.add_argument('--train_batch_size', type=int,default=2, help="Batch size for the training dataloader.")
    parser.add_argument('--eval_batch_size', type=int, default=2, help="Batch size for the evaluation dataloader.")
    parser.add_argument('--preprocessing_batch_size', type=int, default=4, help="Batch size used during the dataset preprocessing phase (in dataloader).")
    parser.add_argument('--padding_side', type=str, default="left", help="Padding side for both tokenizers ('left' or 'right'). Use 'left' for last-token-pooling retrievers such as Qwen embedding models.")

    # --- Loss Function Args ---
    parser.add_argument('--lambda_loss', type=float, default=0.2, help="Weighting factor (lambda) for the ORPO preference loss term.")
    parser.add_argument('--n_neg_examples', type=int, default=3, help="Number of negative samples per positive sample during training.")
    parser.add_argument('--gamma', type=float, default=1, help="Temperature scaling factor (gamma) for retrieval similarities (Pr_retr calculation).")
    parser.add_argument('--beta', type=float, default=1, help="Temperature scaling factor (beta) for perplexities (Q calculation).")
    parser.add_argument('--preference_weight', type=float, default=0.1, help="Weighting factor (beta) within the ORPO odds ratio calculation.")

    # --- Reproducibility & Logging Args ---
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--wandb_project_name', type=str, default="PortsAAAI", help="Weights & Biases project name.")
    parser.add_argument('--wandb_run_name', type=str, default="", help="Weights & Biases run name (defaults to generated name if empty).")
    parser.add_argument('--log_freq', type=int, default=100, help="Log training metrics to W&B every N steps.")

    args = parser.parse_args()

    # ******************** Initial Setup ********************
    set_seed(args.seed) # Set seed for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Parsed arguments: {vars(args)}") # Log parsed arguments

    # ******************** Load Retrieval Model & Tokenizer ********************
    logger.info(f"Loading Retrieval Model: {args.retrieval_model_name}")
    retr_model_name = args.retrieval_model_name
    retr_tokenizer = AutoTokenizer.from_pretrained(retr_model_name)

    retr_quant_config = None
    if args.use_qlora:
        if not args.use_lora:
            raise ValueError("--use_qlora requires --use_lora to be set.")
        retr_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    retr_model = AutoModel.from_pretrained(
        retr_model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=retr_quant_config,
        device_map=device if retr_quant_config is not None else None,
    )
    if retr_quant_config is None:
        retr_model = retr_model.to(device)

    if args.use_lora:
        logger.info(f"Wrapping retrieval model with LoRA (r={args.lora_r}, alpha={args.lora_alpha}, targets={args.lora_target_modules})")
        retr_model.gradient_checkpointing_enable()
        retr_model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            task_type="FEATURE_EXTRACTION",
        )
        retr_model = get_peft_model(retr_model, lora_config)
        retr_model.print_trainable_parameters()

    # ******************** Load Generative Model & Tokenizer ********************
    logger.info(f"Loading Generative Model (Pseudo-Name): {args.inference_model_name}")
    pseudo_model_name = args.inference_model_name
    infer_model_name = pseudo_name_mapping[pseudo_model_name] # Get actual HF name
    logger.info(f"Actual Inference Model Path: {infer_model_name}")
    
    # Using Unsloth for faster and more efficient model loading
    logger.info(f"Loading inference model with Unsloth (4-bit: {args.load_in_4bit})")
    infer_model, infer_tokenizer = FastLanguageModel.from_pretrained(
        model_name=infer_model_name,
        max_seq_length=args.inference_max_seq_length,
        load_in_4bit=args.load_in_4bit,
        device_map=device,
    )
    FastLanguageModel.for_inference(infer_model)
    logger.info(f"Successfully loaded inference model with Unsloth")

    # ******************** Configure Tokenizers (Padding) ********************
    # --- Inference Tokenizer ---
    if infer_tokenizer.pad_token is None:
        logger.warning("Inference tokenizer missing pad token. Adding '<pad>' and using EOS token ID.")
        infer_tokenizer.add_special_tokens({'pad_token': '<pad>'}) 
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
        infer_tokenizer.pad_token_id = infer_tokenizer.eos_token_id
        # No need to explicitly resize embeddings for Unsloth models
        logger.info(f"Inference tokenizer pad_token_id set to: {infer_tokenizer.pad_token_id}")

    infer_tokenizer.padding_side = args.padding_side
    logger.info(f"Inference tokenizer padding side set to: {args.padding_side}")

    # --- Retrieval Tokenizer ---
    if retr_tokenizer.pad_token is None:
        logger.warning("Retrieval tokenizer missing pad token. Adding '<pad>' and using EOS token ID.")
        retr_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        retr_tokenizer.pad_token = retr_tokenizer.eos_token
        retr_tokenizer.pad_token_id = retr_tokenizer.eos_token_id
        retr_model.resize_token_embeddings(len(retr_tokenizer))
        logger.info(f"Retrieval tokenizer pad_token_id set to: {retr_tokenizer.pad_token_id}")

    retr_tokenizer.padding_side = args.padding_side
    logger.info(f"Retrieval tokenizer padding side set to: {args.padding_side}")

    # ******************** Load and Prepare Dataset ********************
    logger.info(f"Loading dataset: {args.dataset}")
    dataset_downloader = DatasetDownloader(dataset_name=args.dataset)
    dataset = dataset_downloader.get_dataset() # Load raw dataset


    print("********************")
    print(dataset)
    print("********************")

    # --- Handle Multi-API Datasets (Special Case) ---
    if args.dataset in ["apibench", "toolbench"]:
        logger.info(f"Processing multi-API setup for dataset: {args.dataset}")
        global query_id_dict, apis_multi
        unique_queries = list(set(dataset["test"]["query_for_retrieval"]))
        query_id_dict = {query: i for i, query in enumerate(unique_queries)}
        apis_multi = {}
        for query, query_id in query_id_dict.items():
             relevant_apis = set(
                 item["api_description"] 
                 for item in dataset["test"] 
                 if item["query_for_retrieval"] == query
             )
             apis_multi[query_id] = list(relevant_apis)
        logger.info(f"Created query_id_dict ({len(query_id_dict)} entries) and apis_multi ({len(apis_multi)} entries)")

    dataset = dataset_downloader.post_process_answers(dataset)

    if args.max_train_samples:
        n_inst = min(args.max_train_samples, len(dataset["train"]))
        logger.info(f"Sampling {n_inst} instances from the training set.")
        selected_indices = random.sample(range(len(dataset["train"])), n_inst)
        dataset["train"] = dataset["train"].select(selected_indices)

    if args.max_eval_samples:
        eval_split_name = "test" if "test" in dataset else "validation"
        if eval_split_name in dataset:
            n_inst = min(args.max_eval_samples, len(dataset[eval_split_name]))
            logger.info(f"Sampling {n_inst} instances from the {eval_split_name} set.")
            selected_indices = random.sample(range(len(dataset[eval_split_name])), n_inst)
            dataset[eval_split_name] = dataset[eval_split_name].select(selected_indices)
        else:
             logger.warning(f"Cannot sample evaluation set: Neither 'test' nor 'validation' split found.")

    logger.info(">>> Final Dataset Stats <<<")
    for split_name in dataset:
        logger.info(f"  Split '{split_name}': {len(dataset[split_name])} examples")

    # ******************** Define API Corpora ********************
    logger.info("Defining training and evaluation API corpora")
    train_api_corpus = list(set(dataset["train"]["api_description"]))
    eval_split_name_for_corpus = "test" if "test" in dataset else "validation"
    if eval_split_name_for_corpus in dataset:
        # Align with train_mnrl.py: eval_api_corpus should be strictly from the eval split.
        eval_api_corpus = list(set(dataset[eval_split_name_for_corpus]["api_description"]))
    else:
        logger.warning(f"No 'test' or 'validation' split found for eval corpus. Using training corpus for evaluation.")
        eval_api_corpus = train_api_corpus # Fallback, though ideally an eval split should exist

    logger.info(f"Corpus sizes: Train: {len(train_api_corpus)} | Eval: {len(eval_api_corpus)}")

    # ******************** Setup Prompting and Collator ********************
    logger.info("Setting up prompt templates and data collator")
    prompt_config = PROMPT_TEMPLATES[pseudo_model_name]
    instruction = pseudo_name_instr_mapping[pseudo_model_name]
    prompt_template_str = prompt_config["prompt_template"]
    answer_template = prompt_config["answer_template"]

    response_template_ids = infer_tokenizer.encode(answer_template,
                                                    add_special_tokens=False)

    data_collator_completion = DataCollatorForCompletionOnlyLM(
        tokenizer=infer_tokenizer, 
        response_template=response_template_ids,
        mlm=False
    )

    # ******************** Prepare Training Arguments ********************
    # Priority: n_reembedding_steps takes precedence over embedding_update_steps
    if args.n_reembedding_steps is not None:
        logger.info(f"Using n_reembedding_steps={args.n_reembedding_steps} (ignoring embedding_update_steps={args.embedding_update_steps})")
        effective_n_reembedding_steps = args.n_reembedding_steps
        embedding_update_steps = None
    elif args.embedding_update_steps is not None:
        # Calculate n_reembedding_steps if only embedding_update_steps is provided
        n_reembedding_steps_calculated = len(dataset["train"]) // args.embedding_update_steps if args.embedding_update_steps > 0 else None
        if n_reembedding_steps_calculated == 0:  # Ensure at least 1 split if dataset is smaller than update_steps
            n_reembedding_steps_calculated = 1
        logger.info(f"Using calculated n_reembedding_steps={n_reembedding_steps_calculated} based on embedding_update_steps={args.embedding_update_steps}")
        effective_n_reembedding_steps = n_reembedding_steps_calculated
        embedding_update_steps = args.embedding_update_steps
    else:
        logger.info("No embedding update strategy specified. Using default.")
        effective_n_reembedding_steps = None
        embedding_update_steps = None
        
    train_eval_config = {
        "dataset" : dataset,
        "dataset_name" : args.dataset,
        "retr_tokenizer" : retr_tokenizer, 
        "retr_model" : retr_model,
        "infer_tokenizer" : infer_tokenizer,
        "infer_model" : infer_model,
        "train_api_corpus" : train_api_corpus,
        "eval_api_corpus" : eval_api_corpus,
        "data_collator_completion" : data_collator_completion,
        "eval_strategy" : args.eval_strategy,
        "eval_steps" : args.eval_steps,
        "n_reembedding_steps" : effective_n_reembedding_steps,
        "embedding_update_steps": embedding_update_steps,
        "prompt_template" : prompt_template_str,
        "instruction_prompt" : instruction,
        "lambda_loss" : args.lambda_loss,
        "beta" : args.beta,
        "gamma" : args.gamma,
        "preference_weight" : args.preference_weight,
        "num_epochs" : args.n_epochs,
        "retriever_max_seq_length" : args.retriever_max_seq_length,
        "inference_max_seq_length" : args.inference_max_seq_length,
        "number_of_neg_examples" : args.n_neg_examples,
        "train_batch_size" : args.train_batch_size,
        "eval_batch_size" : args.eval_batch_size,
        "preprocessing_batch_size" : args.preprocessing_batch_size,
        "learning_rate" : args.lr,
        "scheduler_type" : args.lr_type,
        "warmup_ratio" : args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "log_freq" :  args.log_freq,
        "k_eval_values_accuracy" : args.k_eval_values_accuracy,
        "k_eval_values_ndcg" : args.k_eval_values_ndcg,
        "device" : device,
        "wandb_project_name" : args.wandb_project_name,
        "wandb_run_name" : args.wandb_run_name,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_dir": args.save_dir,
        "max_checkpoints": args.max_checkpoints,
        "save_checkpoints": args.save_checkpoints,
        "infer_model_name": pseudo_model_name,
        "retr_model_name": retr_model_name,
    }

    logger.info("Starting Training and Evaluation Process...")
    train(**train_eval_config)
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()