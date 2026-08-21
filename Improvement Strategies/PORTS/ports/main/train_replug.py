import unsloth
from unsloth import FastLanguageModel

import os
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    get_scheduler, 
    set_seed
)
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from src.port.prompt_templates import (
    pseudo_name_mapping,
    pseudo_name_instr_mapping,
    PROMPT_TEMPLATES
)
import logging
from sentence_transformers import SentenceTransformer, models

import sys
import os
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
import math
import gc  # Import gc for garbage collection

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

# Set up environment
load_dotenv()
app_path = os.environ.get("APP_PATH", os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(app_path))

from src.port.retrieval_evaluator import DeviceAwareInformationRetrievalEvaluator
from src.port.utils import embed_corpus
from src.port.dataset_helper import DatasetDownloader


def main():
    parser = argparse.ArgumentParser(description="RePlug training")
    parser.add_argument('--dataset',                      type=str,   required=True, help="dataset name for DatasetDownloader")
    parser.add_argument('--retr_model_name_or_path',     type=str,   required=True)
    parser.add_argument('--infer_model_name_or_path',    type=str,   required=True)
    parser.add_argument('--infer_model_type',            type=str,   required=True)
    parser.add_argument('--inference_max_seq_length',    type=int,   default=1024, help="Max sequence length for inference model")
    parser.add_argument('--quantize',                    action='store_true')
    parser.add_argument('--quantization_4bit',           action='store_true')
    parser.add_argument('--batch_size',                  type=int,   default=2)
    parser.add_argument('--num_train_epochs',            type=int,   default=5)
    parser.add_argument('--num_retrieved_docs_per_query',type=int,   default=1)
    parser.add_argument('--gamma_value',                 type=float, default=1.0)
    parser.add_argument('--beta_value',                  type=float, default=1.0)
    parser.add_argument('--learning_rate',               type=float, default=1e-5)
    parser.add_argument('--lr_scheduler',                type=str,   default='linear')
    parser.add_argument('--log_to_wandb',                action='store_true')
    parser.add_argument('--wandb_proj_name',             type=str,   default='')
    parser.add_argument('--verbose',                     action='store_true')
    parser.add_argument('--eval_steps',                  type=float, default=None, help="fraction of steps for evaluation")
    parser.add_argument('--k_eval_values_accuracy',     nargs='+', type=int, default=[1,3,5])
    parser.add_argument('--k_eval_values_ndcg',         nargs='+', type=int, default=[1,3,5])
    parser.add_argument('--seed',                        type=int,   default=42)
    parser.add_argument('--trained_model_save_path',     type=str,   default=None)
    parser.add_argument('--max_train_samples',           type=int,   default=None)
    parser.add_argument('--max_eval_samples',            type=int,   default=None)
    parser.add_argument('--retr_max_seq_length',         type=int,   default=512)
    parser.add_argument('--warmup_ratio',                type=float, default=0.1, help="fraction of total training steps for warmup")
    parser.add_argument('--save_strategy',               type=str,   default='epoch')
    parser.add_argument('--save_dir',                    type=str,   default='/workspace/output')
    parser.add_argument('--corpus_updates',              type=int,   default=5, help="Number of corpus embedding updates per epoch")
    parser.add_argument('--preprocess_batch_size',       type=int,   default=16, help="Batch size for preprocessing operations")
    parser.add_argument('--save_checkpoints',            action='store_true', default=False, help="Whether to save model checkpoints during training")
    args = parser.parse_args()

    log_wandb = args.log_to_wandb
    verbose = args.verbose

    def log_mem_usage(topic):
        if verbose:  # Only log memory usage when verbose flag is set
            current_device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(current_device)
            memory_reserved = torch.cuda.memory_reserved(current_device)
            logger.info(f"{topic} A: {memory_allocated / 1024**2:.1f}MB, R: {memory_reserved / 1024**2:.1f}MB")

    set_seed(args.seed)
    retr_model_name = args.retr_model_name_or_path
    infer_model_name = args.infer_model_name_or_path
    retr_tokenizer = AutoTokenizer.from_pretrained(retr_model_name)
    retr_model_base = AutoModel.from_pretrained(retr_model_name).to("cuda")
    
    # Using Unsloth for faster and more efficient model loading
    logger.info(f"Loading inference model with Unsloth (4-bit: {args.quantization_4bit})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer_model, infer_tokenizer = FastLanguageModel.from_pretrained(
        model_name=infer_model_name,
        max_seq_length=args.inference_max_seq_length,
        load_in_4bit=args.quantization_4bit,
        device_map=device,
    )
    FastLanguageModel.for_inference(infer_model)
    logger.info(f"Successfully loaded inference model with Unsloth")

    if infer_tokenizer.pad_token is None:
        logger.info("No padding token - using EOS instead")
        infer_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
        infer_tokenizer.pad_token_id = infer_tokenizer.eos_token_id
        logger.info(f"Inference tokenizer pad_token_id set to: {infer_tokenizer.pad_token_id}")
    
    infer_tokenizer.padding_side = 'left'
    logger.info(f"Inference tokenizer padding side set to: {infer_tokenizer.padding_side}")

    eval_steps_fraction = getattr(args, 'eval_steps', None)

    # Get k_eval values for evaluation
    k_values_accuracy = getattr(args, 'k_eval_values_accuracy', [1, 3, 5])
    k_values_ndcg = getattr(args, 'k_eval_values_ndcg', [1, 3, 5])

    if log_wandb:
        run_name = args.wandb_proj_name or f"REPLUG-{args.dataset}-{args.num_train_epochs}ep"
        logger.info(f"Initializing W&B with run name: {run_name}")
        wandb_config = {
            "retr_model_name_or_path": args.retr_model_name_or_path,
            "infer_model_name_or_path": args.infer_model_name_or_path,
            "quantize": args.quantize,
            "quantization_4bit": args.quantization_4bit,
            "batch_size": args.batch_size,
            "num_train_epochs": args.num_train_epochs,
            "num_retrieved_docs_per_query": args.num_retrieved_docs_per_query,
            "gamma_value": args.gamma_value,
            "beta_value": args.beta_value,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler,
            "dataset": args.dataset,
            "warmup_ratio": args.warmup_ratio,
            "corpus_updates": args.corpus_updates,
            "preprocess_batch_size": args.preprocess_batch_size
        }
        if eval_steps_fraction is not None:
            wandb_config["eval_steps_fraction"] = eval_steps_fraction

        wandb.init(
            project=os.getenv('WANDB_PROJECT_NAME', 'REPLUG_Training'),
            name=run_name,
            config=wandb_config
        )
        log_freq = getattr(args, 'log_freq', 100)  # Default to 100 if not provided
        wandb.watch(retr_model_base, log_freq=log_freq)

    infer_model.eval()
    retr_model_base.eval()

    # Map pseudo-name to actual model path and type
    infer_model_type = args.infer_model_type
    
    # Get prompt template and instruction the same way as in main_train_port.py
    # Find the pseudo-name key that corresponds to this model type
    pseudo_model_name = None
    for key, value in pseudo_name_mapping.items():
        if value == infer_model_name or key == infer_model_type:
            pseudo_model_name = key
            break
    
    if not pseudo_model_name:
        logger.warning(f"Could not find matching pseudo_model_name for {infer_model_name}. Using infer_model_type directly.")
        pseudo_model_name = infer_model_type
    
    prompt_config = PROMPT_TEMPLATES[pseudo_model_name]
    instruction = pseudo_name_instr_mapping[pseudo_model_name]
    prompt_template = prompt_config["prompt_template"]
    ANSWER = prompt_config["answer_template"]
    
    logger.info(f"Using prompt template and instruction for model: {pseudo_model_name}")
    
    # Load dataset using the DatasetDownloader
    dataset_downloader = DatasetDownloader(dataset_name=args.dataset)
    dataset = dataset_downloader.get_dataset()
    dataset = dataset_downloader.post_process_answers(dataset)

    # Determine evaluation split name
    available_splits = list(dataset.keys())
    logger.info(f"Available splits after processing: {available_splits}")
    
    # If we only have 'train' split, create a test split from it
    if len(available_splits) == 1 and 'train' in available_splits:
        logger.warning("Only 'train' split available. Creating test split from train data.")
        # Create a train/test split (80/20)
        split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
        # Reconstruct the dataset with both splits
        dataset = DatasetDict({
            'train': split_dataset['train'],
            'test': split_dataset['test']
        })
        available_splits = list(dataset.keys())
        logger.info(f"Created splits: {available_splits}")

    # Now determine which split to use for evaluation queries (prioritize validation for dev set)
    if 'validation' in dataset:
        eval_split_name = 'validation'
        logger.info(f"Using 'validation' split as the primary evaluation split (for queries and as corpus fallback).")
    elif 'test' in dataset:
        eval_split_name = 'test'
        logger.info(f"Using 'test' split as the primary evaluation split (for queries and as corpus fallback), 'validation' not found.")
    else:
        available_splits = list(dataset.keys())
        raise ValueError(f"Could not find 'validation' or 'test' split for evaluation. Available splits: {available_splits}")
    
    logger.info(f"Dataset split sizes: Train={len(dataset['train'])}, {eval_split_name}={len(dataset[eval_split_name])}")

    # Sample data if max_train_samples or max_eval_samples is set
    if getattr(args, "max_train_samples", None):
        n_inst = min(args.max_train_samples, len(dataset["train"]))
        selected_indices = np.random.choice(len(dataset["train"]), n_inst, replace=False)
        dataset["train"] = dataset["train"].select(selected_indices)

    if getattr(args, "max_eval_samples", None):
        n_inst = min(args.max_eval_samples, len(dataset[eval_split_name]))
        logger.info(f"Sampling {n_inst} instances from {eval_split_name} split.")
        selected_indices = np.random.choice(len(dataset[eval_split_name]), n_inst, replace=False)
        dataset[eval_split_name] = dataset[eval_split_name].select(selected_indices)

    # Define API documents consistently
    train_api_corpus = list(set(dataset["train"]["api_description"]))
    
    # Determine the source for the evaluation API corpus.
    # Prioritize 'validation' split if it exists, for "dev test only" evaluation.
    if 'validation' in dataset:
        logger.info("Using 'validation' split for the evaluation API corpus.")
        eval_api_corpus = list(set(dataset['validation']['api_description']))
    elif eval_split_name in dataset: # Fallback to eval_split_name (e.g., 'test') if 'validation' not found
        logger.info(f"Using '{eval_split_name}' split for the evaluation API corpus ('validation' split not found).")
        eval_api_corpus = list(set(dataset[eval_split_name]['api_description']))
    else:
        # This case should ideally not be reached if eval_split_name is correctly determined earlier.
        logger.error(f"Cannot determine evaluation API corpus. Neither 'validation' nor '{eval_split_name}' found in dataset splits: {list(dataset.keys())}.")
        logger.warning("Falling back to using 'train_api_corpus' for evaluation API corpus due to missing dedicated eval corpus source.")
        eval_api_corpus = train_api_corpus # Fallback to train corpus if no suitable eval corpus found

    logger.info(f"Corpus sizes: Train={len(train_api_corpus)}, Eval={len(eval_api_corpus)}")

    # Update tokenize function to match main_train_port.py style
    def tokenize_function(examples):
        # Process queries in batch mode
        tokenized = retr_tokenizer(
            examples["query"],
            padding="max_length",
            truncation=True,
            max_length=args.retr_max_seq_length,
            return_tensors="pt"
        )
        return tokenized

    # Determine number of processes for parallel mapping
    num_proc = min(os.cpu_count() or 1, 8)  # Limit to 8 processes max
    logger.info(f"Using {num_proc} processes for dataset mapping")

    input_training_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        batch_size=args.preprocess_batch_size,
        num_proc=num_proc,
        remove_columns=dataset["train"].column_names
    )

    input_eval_dataset = dataset[eval_split_name].map(
        tokenize_function,
        batched=True,
        batch_size=args.preprocess_batch_size,
        num_proc=num_proc,
        remove_columns=dataset[eval_split_name].column_names
    )

    batch_size = args.batch_size
    num_epochs = args.num_train_epochs
    k = args.num_retrieved_docs_per_query
    gamma = args.gamma_value
    beta = args.beta_value
    data_collator = DataCollatorWithPadding(tokenizer=retr_tokenizer)
    train_data_loader = DataLoader(
        input_training_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
    )
    eval_data_loader = DataLoader(
        input_eval_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
    )

    optimizer = AdamW(retr_model_base.parameters(), lr=args.learning_rate)
    num_training_steps = num_epochs * len(train_data_loader)
    warmup_ratio = getattr(args, 'warmup_ratio', 0.1)
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    logger.info(f"Training setup: Steps={num_training_steps}, Warmup={num_warmup_steps}, LR={args.learning_rate}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    cross_entropy = CrossEntropyLoss(reduction='none')
    kl_div = KLDivLoss(reduction='none')

    inner_data_collator = DataCollatorForCompletionOnlyLM(
        response_template=ANSWER, 
        tokenizer=infer_tokenizer, 
        mlm=False)

    def inner_tokenize_function(examples):
        # Apply truncation and padding in batch mode
        return infer_tokenizer(
            examples["text"], 
            truncation=True, 
            padding=True, 
            max_length=args.inference_max_seq_length,  # Explicitly set the max length here
            return_tensors="pt"
        )
    
    def parse_batch(dataset_split, batch, index):
        curr_bs = batch["input_ids"].size(0)
        batch_data = dataset_split[index*batch_size:(index*batch_size)+curr_bs]
        return curr_bs, batch_data

    def compute_Pr(similarities, gamma):
        # Ensure tensor requires gradients
        if not similarities.requires_grad:
            similarities = similarities.detach().requires_grad_(True)
        return F.softmax(similarities / gamma, dim=1)

    def compute_Q(perplexities, beta):
        perplexities = torch.stack(perplexities).T
        # Ensure tensor requires gradients
        if not perplexities.requires_grad:
            perplexities = perplexities.detach().requires_grad_(True)
        Q = F.softmax(perplexities / beta, dim=1)
        return Q

    def compute_loss(Q, Pr, kl_div):
        # Make sure both tensors require gradients
        if not Q.requires_grad:
            Q = Q.detach().requires_grad_(True)
        if not Pr.requires_grad:
            Pr = Pr.detach().requires_grad_(True)
        Q_log = torch.log(Q)
        divergence = kl_div(Q_log, Pr).sum(-1)
        loss = divergence.mean()
        return loss

    def get_prompts(prompt_template, instruction, documents, batch_data, documents_per_query):
        prompts = [
                prompt_template.format(
                    instruction=instruction, # Use the instruction from pseudo_name_instr_mapping
                    api_def=documents[doc_index], # Map document to api_def
                    query=batch_data["query"][data_index], # Map query
                    answer=batch_data["answer"][data_index] # Map answer
                )
                for i_th_doc in range(documents_per_query.size(1))
                for data_index, doc_index in enumerate(documents_per_query[:, i_th_doc])
            ]
        return prompts

    def prepare_inner_data_loader(prompts, curr_bs, inner_tokenize_function, inner_data_collator):
        inner_dataset = Dataset.from_pandas(pd.DataFrame(prompts, columns=["text"]))
        inner_dataset = inner_dataset.map(
            inner_tokenize_function,
            batched=True,
            batch_size=args.preprocess_batch_size,
            num_proc=num_proc,  # Add parallel processing
            remove_columns=inner_dataset.column_names
        )
        inner_data_loader = DataLoader(
            inner_dataset, shuffle=False, batch_size=curr_bs, collate_fn=inner_data_collator
        )
        return inner_data_loader

    def get_perplexity_per_sample(outputs, labels, cross_entropy):
        logits = outputs["logits"]
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        elem_wise_loss = cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_sum_per_sample = elem_wise_loss.view(shift_logits.size(0), shift_logits.size(1)).sum(dim=1)
        num_elems_per_sample = torch.sum(shift_labels.ne(-100), dim=1)
        loss_per_sample = loss_sum_per_sample / num_elems_per_sample 
        loss_per_sample = -loss_per_sample
        if verbose:
            logger.info(f"LOSS PER SAMPLE: {-loss_per_sample}")
        return loss_per_sample

    def evaluate_with_retrieval_evaluator(retr_model_eval: SentenceTransformer,
                                          eval_dataset_split: Dataset,
                                          eval_api_corpus: list,
                                          corpus_embeddings: torch.Tensor,
                                          device: str = "cuda",
                                          k_values_accuracy: list = [1, 3, 5],
                                          k_values_ndcg: list = [1, 3, 5],
                                          eval_batch_size: int = 16,
                                          prefix: str = "eval",
                                          epoch: int = -1,
                                          steps: int = -1):
        logger.info(f"Running {prefix} evaluation")

        queries = {}
        relevant_docs = {}
        corpus = {str(idx): doc for idx, doc in enumerate(eval_api_corpus)}
        api_to_doc_id = {api_desc: str(idx) for idx, api_desc in enumerate(eval_api_corpus)}

        # First, collect all queries and their relevant API descriptions
        query_to_positives = {}
        for i, example in enumerate(eval_dataset_split):
            query_text = example["query"]
            positive_apis = example["api_description"]
            
            if not isinstance(positive_apis, list):
                positive_apis = [positive_apis]
                
            if query_text not in query_to_positives:
                query_to_positives[query_text] = set()
            
            for pos_api in positive_apis:
                if pos_api in api_to_doc_id:
                    query_to_positives[query_text].add(pos_api)

        warning_count = 0
        # Now create the actual queries and relevant_docs structure
        for i, example in enumerate(eval_dataset_split):
            query_text = example["query"]
            positive_apis = example["api_description"]
            
            if not isinstance(positive_apis, list):
                positive_apis = [positive_apis]

            query_id = f"query_{i}"
            queries[query_id] = query_text
            
            current_relevant_ids = set()
            for pos_api in query_to_positives.get(query_text, set()):
                 if pos_api in api_to_doc_id:
                     current_relevant_ids.add(api_to_doc_id[pos_api])
                 else:
                      warning_count += 1
                      if warning_count <= 3:
                          logger.warning(f"Missing API: '{pos_api[:30]}...'")
                      elif warning_count == 4:
                          logger.warning(f"Suppressing further missing API warnings")
            
            if current_relevant_ids:
                 relevant_docs[query_id] = current_relevant_ids

        if not queries:
            logger.error("No queries generated for evaluator.")
            return {}
        if not corpus:
            logger.error("Corpus is empty for evaluator.")
            return {}
        if not relevant_docs:
            logger.warning("Relevant docs mapping is empty for evaluator.")

        # Create the evaluator similar to main_train_port.py approach
        logger.info(f"Creating evaluator '{prefix}' with {len(queries)} queries, {len(corpus)} corpus items.")
        evaluator = DeviceAwareInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            batch_size=eval_batch_size,
            name=prefix,
            show_progress_bar=True,
            ndcg_at_k=k_values_ndcg,
            accuracy_at_k=k_values_accuracy
        )

        # Call the evaluator directly like in main_train_port.py, instead of using compute_metrices
        scores = evaluator(retr_model_eval)

        logger.info(f"{prefix.capitalize()} results at epoch {epoch}, step {steps}:")
        
        key_metrics = []
        for k in [1, 3, 5]:
            if f"cosine_accuracy@{k}" in scores:
                key_metrics.append((f"R@{k}", scores[f"cosine_accuracy@{k}"]))
            elif "accuracy@k" in scores.get("cosine", {}) and k in scores["cosine"]["accuracy@k"]:
                key_metrics.append((f"R@{k}", scores["cosine"]["accuracy@k"][k]))
                
            if f"cosine_ndcg@{k}" in scores:
                key_metrics.append((f"NDCG@{k}", scores[f"cosine_ndcg@{k}"]))
            elif "ndcg@k" in scores.get("cosine", {}) and k in scores["cosine"]["ndcg@k"]:
                key_metrics.append((f"NDCG@{k}", scores["cosine"]["ndcg@k"][k]))
        
        metrics_str = ", ".join([f"{name}: {value:.4f}" for name, value in key_metrics])
        logger.info(f"  {metrics_str}")

        if log_wandb and wandb.run:
            log_prefix = f"{prefix}/"
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

            flat_scores = flatten_dict(scores)
            wandb_log_data = {f"{log_prefix}{k}": v for k, v in flat_scores.items()}
            wandb_log_data[f"{prefix}/epoch"] = epoch
            wandb_log_data[f"{prefix}/step"] = steps

            logger.info(f"Logging metrics to W&B under '{log_prefix}' at epoch {epoch}, step {steps}")
            wandb.log(wandb_log_data, step=steps)

        return scores

    def get_sentence_transformer(base_model, device="cuda", pooling_mode="cls"):
        """
        Create a SentenceTransformer wrapper that preserves model state.
        
        This is crucial for ensuring parameter updates in training
        are properly reflected in evaluation.

        Args:
            base_model: The Hugging Face AutoModel to wrap.
            device (str): Device to place the model on.
            pooling_mode (str): Pooling strategy ('cls' or 'mean').
        """
        # Create a proper wrapper architecture
        word_embedding_model = models.Transformer(base_model.config._name_or_path)
        # Copy the state from the base model to ensure continuity
        word_embedding_model.auto_model = base_model
        
        pooling_mode_cls_token = False
        pooling_mode_mean_tokens = False

        if pooling_mode == "cls":
            pooling_mode_cls_token = True
        elif pooling_mode == "mean":
            pooling_mode_mean_tokens = True

        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_cls_token=pooling_mode_cls_token,
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_max_tokens=False
        )
        
        # Add normalization layer for consistent embeddings
        normalize = models.Normalize()
        
        return SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize], device=device)

    def log_to_wandb(score_dict, epoch, steps, prefix="eval"):
        """
        Log metrics to Weights & Biases with proper flattening and organization.
        """
        if not log_wandb or wandb.run is None:
            return
            
        log_prefix = f"{prefix}/" if prefix else ""
        logger.info(f"Logging {prefix} metrics to W&B at epoch {epoch}, step {steps}")

        # Flatten nested dictionaries for better W&B visualization
        def flatten_dict(d, parent_key='', sep='/'):
            items = []
            for k, v in d.items():
                sanitized_k = str(k).replace('@', '_at_')  # Sanitize keys for W&B
                new_key = f"{parent_key}{sep}{sanitized_k}" if parent_key else sanitized_k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, (int, float, np.number)):
                    items.append((new_key, float(v)))
            return dict(items)

        flat_scores = flatten_dict(score_dict)
        wandb_log_data = {f"{log_prefix}{k}": v for k, v in flat_scores.items()}
        wandb_log_data[f"{prefix}/epoch"] = epoch
        wandb_log_data[f"{prefix}/step"] = steps

        # Log key metrics to console for visibility
        key_metrics = []
        for k in k_values_accuracy:
            if f"cosine_accuracy@{k}" in flat_scores:
                key_metrics.append((f"R@{k}", flat_scores[f"cosine_accuracy@{k}"]))
        for k in k_values_ndcg:
            if f"cosine_ndcg@{k}" in flat_scores:
                key_metrics.append((f"NDCG@{k}", flat_scores[f"cosine_ndcg@{k}"]))
        
        if key_metrics:
            metrics_str = ", ".join([f"{name}: {value:.4f}" for name, value in key_metrics])
            logger.info(f"  {metrics_str}")
        
        wandb.log(wandb_log_data, step=steps)

    steps_per_epoch = len(train_data_loader)
    evaluation_step_interval = None
    run_end_of_epoch_eval = True
    best_eval_score = 0.0

    if eval_steps_fraction is not None and eval_steps_fraction > 0:
        if not (0 < eval_steps_fraction <= 1):
            raise ValueError("eval_steps must be a float between 0 (exclusive) and 1 (inclusive)")
        evaluation_step_interval = max(1, int(steps_per_epoch * eval_steps_fraction))
        logger.info(f"Evaluation strategy: 'steps'. Evaluating every {evaluation_step_interval} steps ({eval_steps_fraction*100:.2f}% of {steps_per_epoch} steps per epoch).")
        run_end_of_epoch_eval = False
    else:
        logger.info(f"Evaluation strategy: 'epoch'. Evaluating every {steps_per_epoch} steps (at the end of each epoch).")
        evaluation_step_interval = steps_per_epoch

    logger.info("--- Initial Evaluation ---")
    retr_model_base.eval()
    # Use 'mean' pooling for initial evaluation to match train_mnrl.py and main_train_port.py
    retr_model_eval_instance = get_sentence_transformer(retr_model_base, device="cuda", pooling_mode="cls")
    with torch.no_grad():
        embedded_eval_documents = embed_corpus(retr_model_eval_instance,
                                               retr_tokenizer,
                                               eval_api_corpus,
                                               device="cuda",
                                               batch_size=args.preprocess_batch_size,  # Use preprocessing batch size
                                               max_length=args.retr_max_seq_length)

    evaluate_with_retrieval_evaluator(
        retr_model_eval=retr_model_eval_instance,
        eval_dataset_split=dataset[eval_split_name],
        eval_api_corpus=eval_api_corpus,
        corpus_embeddings=embedded_eval_documents,
        device="cuda",
        k_values_accuracy=k_values_accuracy,
        k_values_ndcg=k_values_ndcg,
        eval_batch_size=args.batch_size,
        prefix="initial_eval",
        epoch=0,
        steps=0
    )
    del embedded_eval_documents, retr_model_eval_instance
    torch.cuda.empty_cache()

    retr_model_base.train()
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0

        # Create data subsplits based on corpus_updates
        data_splits = []
        steps_per_epoch = len(train_data_loader)
        
        if not args.corpus_updates or args.corpus_updates <= 0:
            # Use a single split if corpus_updates is not set
            data_splits = [range(len(train_data_loader))]
        else:
            # Create subsplits for corpus embedding updates
            subsplit_len = steps_per_epoch // args.corpus_updates
            data_subsplits_lens = [subsplit_len for _ in range(args.corpus_updates)]
            # Handle remainder
            if (rest := steps_per_epoch % args.corpus_updates) != 0:
                data_subsplits_lens.append(rest)
            
            start_idx = 0
            for subsplit_len in data_subsplits_lens:
                end_idx = start_idx + subsplit_len
                data_splits.append(range(start_idx, end_idx))
                start_idx = end_idx

        logger.info(f"Created {len(data_splits)} data splits for embedding corpus updates")

        for split_idx, batch_indices in enumerate(data_splits):
            logger.info(f"Processing split {split_idx+1}/{len(data_splits)} with {len(batch_indices)} batches")
            
            # Temporarily set the base model to eval mode for consistent corpus embeddings
            retr_model_base.eval() 
            
            # Create a SentenceTransformer wrapper. It will pick up the eval state of retr_model_base.
            # Pooling mode 'cls' is used here, consistent with other parts.
            retr_model_train_instance = get_sentence_transformer(retr_model_base, device="cuda", pooling_mode="cls")
            
            logger.info(f"Embedding corpus for split {split_idx+1}")
            with torch.no_grad():  # Disable gradients for corpus embedding
                # Clear CUDA cache before embedding
                torch.cuda.empty_cache()
                
                # Process the corpus in smaller chunks if it's large
                MAX_SAFE_CORPUS_SIZE = 1000  # Adjust based on available GPU memory
                if len(train_api_corpus) > MAX_SAFE_CORPUS_SIZE:
                    logger.info(f"Large corpus detected ({len(train_api_corpus)} items). Processing in chunks.")
                    
                    all_embeddings = []
                    chunk_size = MAX_SAFE_CORPUS_SIZE
                    for i in range(0, len(train_api_corpus), chunk_size):
                        chunk = train_api_corpus[i:i+chunk_size]
                        logger.info(f"  Embedding corpus chunk {i//chunk_size + 1}/{math.ceil(len(train_api_corpus)/chunk_size)} ({len(chunk)} items)")
                        
                        chunk_embeddings = embed_corpus(
                            retr_model_train_instance,
                            retr_tokenizer,
                            chunk,
                            device="cuda",
                            batch_size=min(args.preprocess_batch_size, 8),  # Use smaller batch size
                            max_length=args.retr_max_seq_length
                        )
                        all_embeddings.append(chunk_embeddings)
                        # Force garbage collection
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    
                    # Concatenate all chunk embeddings
                    embedded_documents = torch.cat(all_embeddings, dim=0)
                else:
                    # Original code for smaller corpus
                    embedded_documents = embed_corpus(
                        retr_model_train_instance,
                        retr_tokenizer,
                        train_api_corpus,
                        device="cuda",
                        batch_size=args.preprocess_batch_size,
                        max_length=args.retr_max_seq_length
                    )
            
            # Restore the base model to train mode for the subsequent training steps
            retr_model_base.train()

            # Ensure embedded_documents is on CUDA
            if embedded_documents.device.type != "cuda":
                embedded_documents = embedded_documents.to("cuda")

            # Process each batch in the current split
            for relative_idx, batch_idx in enumerate(batch_indices):
                batch = train_data_loader.__iter__().__next__() if relative_idx == 0 else next(train_data_loader.__iter__())
                current_step_in_epoch = batch_idx + 1
                global_step += 1
                curr_bs, batch_data = parse_batch(dataset["train"], batch, batch_idx)

                # Use the base model directly for query encoding to ensure gradient flow
                query_inputs = {k: batch[k].to("cuda") for k in ["input_ids", "attention_mask"]}
                query_outputs = retr_model_base(**query_inputs)
                
                # Extract embeddings using a similar approach as in main_train_port.py
                if hasattr(query_outputs, 'pooler_output') and query_outputs.pooler_output is not None:
                    # BERT and RoBERTa models have pooler_output
                    embedded_queries = query_outputs.pooler_output
                    if verbose:
                        logger.info("Using pooler_output for embeddings")
                elif hasattr(query_outputs, 'last_hidden_state'):
                    # Use CLS token for models without pooler_output
                    embedded_queries = query_outputs.last_hidden_state[:, 0]  # Use first token ([CLS])
                    if verbose:
                        logger.info("Using CLS token from last_hidden_state for embeddings")
                else:
                    # Fallback for other model types
                    logger.warning("Unusual model output format. Using first element of tuple.")
                    embedded_queries = query_outputs[0][:, 0]  # Try to get first token from first tensor
                
                # Manual similarity calculation with gradients
                embedded_queries = F.normalize(embedded_queries, p=2, dim=1)
                embedded_documents_normalized = F.normalize(embedded_documents, p=2, dim=1)
                
                cos_sim = torch.matmul(embedded_queries, embedded_documents_normalized.T)
                
                # Get top k documents
                top_k_docs = torch.topk(cos_sim, k, dim=-1)
                documents_per_query = top_k_docs.indices
                similarities_per_query = top_k_docs.values
                
                # Compute Pr with gradients
                Pr = compute_Pr(similarities_per_query, gamma)
                
                # Clean up to save memory
                del query_outputs
                torch.cuda.empty_cache()
                
                prompts = get_prompts(prompt_template, instruction, train_api_corpus, batch_data, documents_per_query)
                inner_data_loader = prepare_inner_data_loader(prompts, curr_bs, inner_tokenize_function, inner_data_collator)
                perplexities = []
                for inner_batch in inner_data_loader:
                    inner_batch = {k: v.to("cuda") for k, v in inner_batch.items()}
                    labels = inner_batch.pop("labels")
                    
                    with torch.no_grad():
                        outputs = infer_model(**inner_batch)
                    perplexity = get_perplexity_per_sample(outputs, labels, cross_entropy)
                    perplexities.append(perplexity)
                    del outputs, perplexity
                    torch.cuda.empty_cache()
                Q = compute_Q(perplexities, beta)

                loss = compute_loss(Q, Pr, kl_div)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                assert not torch.isnan(loss), "Loss is NaN"
                assert loss < 1e6, f"Loss is too large: {loss}"
                del perplexities, Q, Pr, inner_data_loader, inner_batch, documents_per_query
                torch.cuda.empty_cache()

                epoch_loss += loss.item()
                batch_count += 1

                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(retr_model_base.parameters(), max_norm=1.0)
                
                # Enhanced logging for better visibility
                if global_step % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs} | Step {global_step} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

                if log_wandb and global_step % min(20, max(1, len(train_data_loader)//5)) == 0:
                    wandb.log({
                        "train/replug_loss": loss.item(),
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "progress/epoch": epoch + 1,
                    }, step=global_step)

                # Step-based evaluation with proper model handling
                if evaluation_step_interval is not None and global_step % evaluation_step_interval == 0:
                    logger.info(f"--- Evaluating at Epoch {epoch+1}, Step {current_step_in_epoch}/{steps_per_epoch} (Global Step {global_step}) ---")
                    retr_model_base.eval()
                    retr_model_eval_instance = get_sentence_transformer(retr_model_base, device="cuda")
                    with torch.no_grad():
                        embedded_eval_documents = embed_corpus(retr_model_eval_instance,
                                                               retr_tokenizer,
                                                               eval_api_corpus,
                                                               device="cuda",
                                                               batch_size=args.batch_size,
                                                               max_length=args.retr_max_seq_length)

                    eval_scores = evaluate_with_retrieval_evaluator(
                        retr_model_eval=retr_model_eval_instance,
                        eval_dataset_split=dataset[eval_split_name],
                        eval_api_corpus=eval_api_corpus,
                        corpus_embeddings=embedded_eval_documents,
                        device="cuda",
                        k_values_accuracy=k_values_accuracy,
                        k_values_ndcg=k_values_ndcg,
                        eval_batch_size=args.batch_size,
                        prefix="eval",
                        epoch=epoch + 1,
                        steps=global_step
                    )
                    
                    # Track best model for potential checkpointing
                    current_score = None
                    for k in [1]:  # Use R@1 as primary metric
                        score_key = f"cosine_accuracy@{k}"
                        if score_key in eval_scores:
                            current_score = eval_scores[score_key]
                            break
                    
                    if current_score and current_score > best_eval_score:
                        best_eval_score = current_score
                        if args.save_checkpoints:
                            best_model_path = os.path.join(args.save_dir, "best_model")
                            os.makedirs(best_model_path, exist_ok=True)
                            retr_model_base.save_pretrained(best_model_path)
                            retr_tokenizer.save_pretrained(best_model_path)
                            logger.info(f"New best model saved with score {current_score:.4f}")
                    
                    retr_model_base.train()
                    del embedded_eval_documents, retr_model_eval_instance
                    torch.cuda.empty_cache()

            # Clean up at the end of each split
            del embedded_documents, retr_model_train_instance
            torch.cuda.empty_cache()

        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed, avg loss: {avg_epoch_loss:.4f}")

        if run_end_of_epoch_eval:
            if evaluation_step_interval is None or global_step % evaluation_step_interval != 0:
                logger.info(f"--- Evaluating after Epoch {epoch+1} (Global Step {global_step}) ---")
                retr_model_base.eval()
                retr_model_eval_instance = get_sentence_transformer(retr_model_base, device="cuda")
                with torch.no_grad():
                    embedded_eval_documents = embed_corpus(retr_model_eval_instance,
                                                           retr_tokenizer,
                                                           eval_api_corpus,
                                                           device="cuda",
                                                           batch_size=args.preprocess_batch_size,  # Use preprocessing batch size
                                                           max_length=args.retr_max_seq_length)

                eval_scores = evaluate_with_retrieval_evaluator(
                    retr_model_eval=retr_model_eval_instance,
                    eval_dataset_split=dataset[eval_split_name],
                    eval_api_corpus=eval_api_corpus,
                    corpus_embeddings=embedded_eval_documents,
                    device="cuda",
                    k_values_accuracy=k_values_accuracy,
                    k_values_ndcg=k_values_ndcg,
                    eval_batch_size=args.batch_size,
                    prefix="eval",
                    epoch=epoch + 1,
                    steps=global_step
                )
                
                # Track best model for potential checkpointing
                current_score = None
                for k in [1]:  # Use R@1 as primary metric
                    score_key = f"cosine_accuracy@{k}"
                    if score_key in eval_scores:
                        current_score = eval_scores[score_key]
                        break
                
                if current_score and current_score > best_eval_score:
                    best_eval_score = current_score
                    if args.save_checkpoints:
                        best_model_path = os.path.join(args.save_dir, "best_model")
                        os.makedirs(best_model_path, exist_ok=True)
                        retr_model_base.save_pretrained(best_model_path)
                        retr_tokenizer.save_pretrained(best_model_path)
                        logger.info(f"New best model saved with score {current_score:.4f}")
                
                retr_model_base.train()
                del embedded_eval_documents, retr_model_eval_instance
                torch.cuda.empty_cache()

    logger.info(f"Training complete. Best evaluation score: {best_eval_score:.4f}")
    if log_wandb:
        wandb.finish()
    if args.save_checkpoints and args.trained_model_save_path:
        retr_model_base.save_pretrained(args.trained_model_save_path)
        retr_tokenizer.save_pretrained(args.trained_model_save_path)
        logger.info(f"Saved trained base model to {args.trained_model_save_path}")

if __name__ == "__main__":
    main()

