import wandb
import math
from tqdm import tqdm
import argparse
import random
import json
import numpy as np

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
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_from_disk

import sys
import os
import wandb
from tqdm import tqdm
from dotenv import load_dotenv

# Set up environment
load_dotenv()
app_path = os.environ.get("APP_PATH", os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(app_path))

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

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
    """Logs metrics to W&B with structured keys."""
    if wandb.run is None:
        logger.warning("wandb.init() has not been called. Skipping wandb.log().")
        return

    log_prefix = f"{prefix}/" if prefix else ""
    logger.info(f"Logging {prefix} metrics to W&B at epoch {epoch}, step {steps}")

    flat_scores = {}

    def flatten_dict(d, parent_key='', sep='/'):
        """Flattens a nested dictionary for W&B logging."""
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

    key_metrics = {}
    for key, value in wandb_log_data.items():
        if any(important in key for important in ['accuracy_at_1', 'ndcg_at_1', 'loss/total', 'epoch', 'step']):
            key_metrics[key] = value
            logger.info(f"  {key}: {value}")
    
    if key_metrics and len(key_metrics) < 5:
        for key, value in wandb_log_data.items():
            if key not in key_metrics and isinstance(value, (int, float)):
                if len(key_metrics) < 10:
                    key_metrics[key] = value
                    logger.info(f"  {key}: {value}")
    
    wandb.log(wandb_log_data, step=steps)


def create_api_evaluator(triplets: list,
                         eval_api_corpus: list,
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

    logger.info(f"Creating evaluator '{name}' with {len(triplets)} triplets / {len(corpus)} corpus items.")

    warning_count = 0
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
            if warning_count <= 3:
                logger.warning(f"API not in corpus: '{positive[:30]}...'")
            elif warning_count == 4:
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

    logger.info(f"Created evaluator '{name}' with {len(queries)} queries, {len(corpus)} corpus docs.")
    return evaluator

def run_evaluation(
    retr_model: AutoModel,
    retr_tokenizer: AutoTokenizer,
    dataset: Dataset,
    eval_api_corpus: list,
    retriever_max_seq_length: int,
    eval_batch_size: int,
    preprocessing_batch_size: int,
    device: str,
    k_eval_values_accuracy: list = [1, 3, 5],
    k_eval_values_ndcg: list = [1, 3, 5],
    dataset_name: str = "toole",
    eval_name: str = "eval",
    epoch: int = 0,
    steps: int = 0
):
    logger.info(f"Running evaluation: {eval_name}")
    eval_split = dataset["train"] if eval_name == "train_eval" else (dataset["test"] if "test" in dataset else dataset["validation"])
    triplets = []
    corpus_set = set(eval_api_corpus)

    print(f"Evaluation split size: {len(eval_split)}")
    print(f"Corpus size: {len(corpus_set)}")
    
    # First, build a mapping of queries to their positive API descriptions
    query_to_positives = {}
    for row in eval_split:
        query = row.get("query")
        positive_api = row.get("api_description")
        if not query or not positive_api or positive_api not in corpus_set:
            continue
        if query not in query_to_positives:
            query_to_positives[query] = set()
        query_to_positives[query].add(positive_api)
    
    logger.info(f"Found {len(query_to_positives)} unique queries with positive examples")

    for row in eval_split:
        query = row.get("query")
        positive_api = row.get("api_description")
        if not query or not positive_api or positive_api not in corpus_set:
            continue
        # Get all positives for this query and exclude them from potential negatives
        query_positives = query_to_positives.get(query, set())
        potential_negatives = list(corpus_set - query_positives)
        if not potential_negatives:
            continue
        negative_api = random.choice(potential_negatives)
        triplets.append((query, positive_api, negative_api))

    if not triplets:
        logger.warning(f"No triplets available for evaluation ({eval_name})")
        return [], [], {}

    logger.info(f"Preparing SentenceTransformer wrapper")
    tmp_model_path = os.path.join("tmp_retr_model_eval")
    retr_model.save_pretrained(tmp_model_path)
    retr_tokenizer.save_pretrained(tmp_model_path)

    transformer = models.Transformer(tmp_model_path)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
        pooling_mode_mean_sqrt_len_tokens=False
    )
    normalize = models.Normalize()
    st_retr_model = SentenceTransformer(modules=[transformer, pooling, normalize], device=device)

    evaluator = create_api_evaluator(
        triplets,
        eval_api_corpus,
        batch_size=eval_batch_size,
        name=eval_name,
        k_values_accuracy=k_eval_values_accuracy,
        k_values_ndcg=k_eval_values_ndcg
    )

    logger.info(f"Computing evaluation scores")
    scores = evaluator(st_retr_model)
    
    if hasattr(run_evaluation, 'first_run') == False:
        logger.info(f"Score structure: {list(scores.keys())}")
        run_evaluation.first_run = True

    log_to_wandb(scores, epoch=epoch, steps=steps, prefix=eval_name)

    ranks = []
    ndcg_scores_list = []
    acc_k_scores = scores.get('cosine', {}).get('accuracy@k', {})
    for k in k_eval_values_accuracy:
        ranks.append(acc_k_scores.get(k, 0.0) * 100)

    ndcg_k_scores = scores.get('cosine', {}).get('ndcg@k', {})
    for k in k_eval_values_ndcg:
        ndcg_scores_list.append(ndcg_k_scores.get(k, 0.0))

    retr_model.train()
    torch.cuda.empty_cache()

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
           prompt_template: str = "",
           instruction_prompt: str = "",
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
           max_checkpoints: int = None):

    config = {
        "beta" : beta,
        "gamma" : gamma,
        "lr" : learning_rate,
        "scheduler_type" : scheduler_type,
        "train_batch_size" : train_batch_size,
        "eval_batch_size" : eval_batch_size,
        "lambda_loss_factor" : lambda_loss,
        "retriever_max_seq_length" : retriever_max_seq_length,
        "inference_max_seq_length" : inference_max_seq_length,
        "epochs" : num_epochs,
        "dataset" : dataset_name,
        "embedding_update_steps" : len(dataset["train"]) // n_reembedding_steps if n_reembedding_steps else "N/A",
        "train_data_samples" : len(dataset["train"])
    }

    run_name = wandb_run_name or f"PORTS-{dataset_name}-{num_epochs}ep"
    logger.info(f"Initializing W&B with run name: {run_name}")
    wandb.init(project=wandb_project_name, name=run_name)
    wandb.config.update(config)
    wandb.watch(retr_model, log_freq=log_freq)

    infer_model.eval()
    
    logger.info(f"Starting initial evaluations")
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
    run_evaluation(**eval_config)

    logger.info(f"Starting Initial Evaluation (Train)")
    train_eval_config = eval_config.copy()
    train_eval_config["eval_api_corpus"] = train_api_corpus
    train_eval_config["eval_name"] = "train_eval"
    run_evaluation(**train_eval_config)

    optimizer = torch.optim.AdamW(
        retr_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    ds_length = len(dataset["train"])
    n_iters = ds_length / train_batch_size + ds_length % train_batch_size
    num_training_steps = num_epochs * n_iters
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    logger.info(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps} ({warmup_ratio*100}%)")

    lr_scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    kl_div = KLDivLoss(reduction='none')

    saved_checkpoints = []

    for epoch in range(num_epochs):
        retr_model.train()
        data_splits = []

        if not n_reembedding_steps:
            data_splits = [dataset]
        else:
            subsplit_len = ds_length // n_reembedding_steps
            data_subsplits_lens = [subsplit_len for _ in range(n_reembedding_steps)]
            if (rest := ds_length % n_reembedding_steps) != 0: data_subsplits_lens.append(rest)

            logger.info("Creating data sub-splits")
            for idx, _len in enumerate(data_subsplits_lens):
                _start = idx * subsplit_len
                _ds_sample = dataset["train"].select(range(_start, _start+_len))
                data_splits.append(DatasetDict({"train":_ds_sample}))

        logger.info(f"Starting training epoch {epoch+1}/{num_epochs}")
        curr_global_step = 0
        epoch_loss = 0.0

        for ds in data_splits:

            logger.info(">> New data split")
            train_data_config = {
                "dataset" : ds,
                "api_corpus_list" : train_api_corpus,
                "retr_model" : retr_model,
                "retrieval_max_length" : retriever_max_seq_length,
                "generateor_max_length" : inference_max_seq_length,
                "retrieval_tokenizer" : retr_tokenizer,
                "inference_tokenizer" : infer_tokenizer,
                "epoch_number" : epoch,
                "batch_size" : train_batch_size,
                "prompt_template" : prompt_template,
                "num_neg_examples" : number_of_neg_examples,
                "preprocessing_batch_size" : preprocessing_batch_size
            }
            triplet_dataloader = get_train_dataloader(**train_data_config)

            steps_per_split = len(triplet_dataloader)
            evaluation_step_interval = None
            if eval_strategy == "steps" and eval_steps is not None:
                if not (0 < eval_steps <= 1):
                    raise ValueError("eval_steps must be a float between 0 (exclusive) and 1 (inclusive) when eval_strategy is 'steps'")
                evaluation_step_interval = max(1, int(steps_per_split * eval_steps))
                logger.info(f"Evaluation strategy: 'steps'. Evaluating every {evaluation_step_interval} steps ({eval_steps*100:.2f}% of {steps_per_split} steps in this split).")

            pbar = tqdm(enumerate(triplet_dataloader), 
                        total=len(triplet_dataloader), 
                        desc=f"Epoch {epoch+1}/{num_epochs}",
                        miniters=max(1, len(triplet_dataloader)//20))
            
            for bid, batch in pbar:
                curr_global_step += 1

                bs = batch["query"]["input_ids"].shape[0]

                pos_labels = batch["labels_pos"]
                neg_labels = batch["labels_neg"]

                queries = batch["query"]
                pos_docs = batch["positive"]
                neg_docs =  batch["negative"]

                n_neg_docs = neg_docs[0]["input_ids"].shape[0]

                pos_simialrity = compute_similarity(retr_model, queries, pos_docs).view(bs,-1)

                neg_similarity = []
                for nid in range(n_neg_docs):
                    data = {
                        k : torch.stack([neg_docs[bid][k][nid,:] for bid in range(bs)]) for k in ['input_ids', 'attention_mask']
                    }
                    this_neg_similarity = compute_similarity(retr_model, 
                                                            queries, 
                                                            data).view(bs,-1)
                    neg_similarity.append(this_neg_similarity)
                
                neg_similarity = torch.stack(neg_similarity, dim=-1)
                similarities = torch.cat((pos_simialrity, neg_similarity.squeeze(1)), dim=-1)

                similarities = similarities / gamma

                Pr_retr = compute_Pr(
                    similarities = similarities,
                    axis = -1
                )

                input_prompt_pos = batch["q_pos_prompt"]
                input_prompt_neg = batch["q_neg_prompt"]

                input_prompt_pos = {k : input_prompt_pos[k].to(device) for k in input_prompt_pos}
                input_prompt_neg = [{k : neg_docs_trip[k].to(device) for k in neg_docs_trip} for neg_docs_trip in input_prompt_neg]

                pos_labels = {k : pos_labels[k].to(device) for k in pos_labels}
                neg_labels = [{k : neg_docs_trip[k].to(device) for k in neg_docs_trip} for neg_docs_trip in neg_labels]

                pos_perplexity = []
                neg_perplexity = []

                with torch.no_grad():
                    pos_data = next(iter(DataLoader(
                        Dataset.from_dict(input_prompt_pos), 
                        shuffle=False, 
                        batch_size=bs, 
                        collate_fn=data_collator_completion)))

                    pos_data = {k: v.to(device) for k, v in pos_data.items()}
                    labels = pos_data.pop("labels")

                    outputs_pos = infer_model(**pos_data)

                    pos_perplexity = get_perplexity(outputs=outputs_pos, 
                                                    input_ids=labels,
                                                    attention_mask=pos_data["attention_mask"],
                                                    padding_token_ids=retr_tokenizer.pad_token_id)
                    del outputs_pos, pos_data
                    torch.cuda.empty_cache()

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

                        neg_perplexity.append(get_perplexity(outputs=outputs_neg, 
                                                            input_ids=labels,
                                                            attention_mask=neg_data["attention_mask"],
                                                            padding_token_ids=retr_tokenizer.pad_token_id))

                        del outputs_neg, neg_data
                        torch.cuda.empty_cache()

                neg_perplexity = torch.stack(neg_perplexity, dim=-1)
                concat_perplexities = torch.cat((pos_perplexity.unsqueeze(0).T, neg_perplexity), dim=-1)
                concat_perplexities = concat_perplexities / beta
                
                Q = F.softmax(concat_perplexities, dim=-1)

                del concat_perplexities
                torch.cuda.empty_cache()

                ppl_pr_KL_loss = compute_loss(Q, Pr_retr, kl_div)

                pref_loss = 0
                pos_rewards, neg_rewards = [], []
                pref_ratio, maean_prob_ratio = 0, 0

                pos_retrieval_prob = Pr_retr[:, 0]
                neg_retrieval_probs = Pr_retr[:, 1:]

                for neg_i in range(n_neg_docs):
                    neg_retrieval_prob = neg_retrieval_probs[:, neg_i]
                    
                    _pref_loss, _pos_reward, _neg_reward, _pref_ratio, _maean_prob_ratio = odds_ratio_loss(
                        positive_retr_log_prob=pos_retrieval_prob.log(),
                        negative_retr_log_prob=neg_retrieval_prob.log(),
                        beta=preference_weight
                    )
                    pref_loss += _pref_loss.mean()
                    pos_rewards.append(_pos_reward.mean())
                    neg_rewards.append(_neg_reward.mean())
                    pref_ratio += _pref_ratio
                    maean_prob_ratio += _maean_prob_ratio

                pref_loss /= n_neg_docs
                pref_ratio /= n_neg_docs
                maean_prob_ratio /= n_neg_docs

                loss = ppl_pr_KL_loss - lambda_loss * pref_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retr_model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                
                grad_norm = get_gradient_norm(retr_model)

                optimizer.zero_grad()
                
                pos_rewards = torch.tensor(pos_rewards)
                neg_rewards = torch.tensor(neg_rewards)
                retrieval_accuracy = (pos_rewards > neg_rewards).float()

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'RePlug': f"{ppl_pr_KL_loss.item():.4f}", 
                    "ORPO": f"{-pref_loss.item():.4f}"
                })
                
                epoch_loss += loss.item()

                if curr_global_step % log_freq == 0:
                    log_metrics = {
                        "loss/total": loss.item(),
                        "loss/replug_kl": ppl_pr_KL_loss.item(),
                        "loss/opro": -pref_loss.item(),
                        "opro/ratio_reward": pref_ratio,
                        "opro/retrieval_accuracy": retrieval_accuracy.mean().cpu(),
                        "opro/mean_prob_ratio": maean_prob_ratio,
                        "probabilities/positive_retrieval": pos_retrieval_prob.mean().cpu(),
                        "probabilities/negative_retrieval": neg_retrieval_probs.mean().cpu(),
                        "log_probabilities/positive": pos_rewards.mean().cpu(),
                        "log_probabilities/negative": neg_rewards.mean().cpu(),
                        "perplexity/positive": pos_perplexity.mean().cpu(),
                        "perplexity/negative": neg_perplexity.mean(-1).mean().cpu(),
                        "Q_values/positive": Q[:,0].mean().cpu(),
                        "Q_values/negative": Q[:,1:].mean(-1).mean().cpu(),
                        "similarity/positive": pos_simialrity.mean().cpu(),
                        "similarity/negative": neg_similarity.mean(-1).mean().cpu(),
                        "optimizer/gradient_norm": grad_norm,
                        "optimizer/learning_rate": optimizer.param_groups[0]['lr'],
                        "progress/epoch": epoch + 1,
                        "progress/step": curr_global_step
                    }
                    wandb.log(log_metrics, step=curr_global_step)

                del Q, Pr_retr, ppl_pr_KL_loss, pref_loss, loss, pos_retrieval_prob, neg_retrieval_probs, neg_retrieval_prob
                del neg_perplexity, pos_perplexity
                del pref_ratio, retrieval_accuracy, pos_rewards, neg_rewards, maean_prob_ratio
                torch.cuda.empty_cache()

                if eval_strategy == "steps" and evaluation_step_interval is not None and curr_global_step % evaluation_step_interval == 0 and curr_global_step != 0:
                    logger.info(f"Starting evaluation (Test/Validation) epoch {epoch+1}/{num_epochs} - step {curr_global_step}")
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
                        "epoch": epoch + 1,
                        "steps": curr_global_step
                    }
                    run_evaluation(**eval_config)

                    logger.info(f"Starting evaluation (Train) epoch {epoch+1}/{num_epochs} - step {curr_global_step}")
                    train_eval_config = eval_config.copy()
                    train_eval_config["eval_api_corpus"] = train_api_corpus
                    train_eval_config["eval_name"] = "train_eval"
                    run_evaluation(**train_eval_config)

                if save_strategy == "steps" and save_steps and curr_global_step % save_steps == 0:
                    logger.info(f"Saving checkpoint at step {curr_global_step}")
                    retr_model.save_pretrained(os.path.join(save_dir, f"checkpoint-step-{curr_global_step}"))

            if save_strategy == "epoch":
                logger.info(f"Saving checkpoint at epoch {epoch+1}")
                ckpt_name = f"checkpoint-epoch-{epoch+1}"
                retr_model.save_pretrained(os.path.join(save_dir, ckpt_name))

                saved_checkpoints.append((ckpt_name, ""))
                if max_checkpoints and len(saved_checkpoints) > max_checkpoints:
                    worst_ckpt = min(saved_checkpoints, key=lambda x: x[1])
                    worst_path = os.path.join(save_dir, worst_ckpt[0])
                    if os.path.exists(worst_path):
                        logger.info(f"Removing worst checkpoint: {worst_ckpt}")
                        import shutil
                        shutil.rmtree(worst_path)
                    saved_checkpoints.remove(worst_ckpt)

        if eval_strategy == "epoch":
            logger.info(f"Starting evaluation (Test/Validation) epoch {epoch+1}/{num_epochs}")
            eval_config = {
                "retr_model" : retr_model,
                "retr_tokenizer" : retr_tokenizer,
                "dataset" : dataset,
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
                "steps": curr_global_step
            }
            run_evaluation(**eval_config)

            logger.info(f"Starting evaluation (Train) epoch {epoch+1}/{num_epochs}")
            train_eval_config = eval_config.copy()
            train_eval_config["eval_api_corpus"] = train_api_corpus
            train_eval_config["eval_name"] = "train_eval"
            run_evaluation(**train_eval_config)

        epoch_avg_loss = epoch_loss / curr_global_step if curr_global_step > 0 else 0
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Avg loss: {epoch_avg_loss:.4f}")

    logger.info("Training and evaluations are over")


def main():
    parser = argparse.ArgumentParser(description='PORT training')
    parser.add_argument('--dataset', type=str, default="bfcl", choices=["bfcl", "apibank", "apibench", "octopus", "octopus-overlap", "toole", "toole-overlap", "toolbench", "toole_90_10", "toole_85_15", "toole_75_25", "toole_70_30", "toole_50_50", "toole_35_65"], help='Dataset name for training and avaluation')

    parser.add_argument('--inference_model_name', type=str, default="llama3-8B", choices=["llama3-8B", "codestral-22B", "gemma2-2B", "groqLlama3Tool-8B"], help="Pseudo-Name of the generative model to use for function calling")
    parser.add_argument('--retrieval_model_name', type=str, default="FacebookAI/roberta-base", help="Name of the encoder model to use for retrieval")
    parser.add_argument('--retriever_max_seq_length', type=int, default=514, help="Max sequence length for retriever")
    parser.add_argument('--inference_max_seq_length', type=int, default=1024, help="Max sequence length for the inference model")

    parser.add_argument('--do_train', action='store_true', default=False, help="Whether to run the training loop")
    parser.add_argument('--do_eval', action='store_true', default=False,  help="Whether to run the evaluation loop")
    parser.add_argument('--load_in_4bit', action='store_true', default=False, help="Whether to load the model in 4 bit")

    parser.add_argument('--eval_strategy', type=str, default="epoch", choices=["epoch", "steps"], help="Strategy to use for evaluation")
    parser.add_argument('--eval_steps', type=float, default=None, help="If eval_strategy='steps', specifies the fraction of steps within an epoch/split after which evaluation is performed (e.g., 0.1 for every 10% of steps).")

    parser.add_argument('--save_strategy', type=str, default="epoch", choices=["epoch", "steps"],
                        help="Strategy to use for saving checkpoints")
    parser.add_argument('--save_steps', type=int, default=None,
                        help="Number of steps after which to save if save_strategy='steps'")
    parser.add_argument('--save_dir', type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument('--max_checkpoints', type=int, default=None,
                        help="Maximum number of checkpoints to store.")

    parser.add_argument('--max_train_samples', type=int, default=None, help="Maximum number of training instances to retain (all if set to None)")
    parser.add_argument('--max_eval_samples', type=int, default=None, help="Maximum number of evaluation instances to retain (all if set to None)")

    parser.add_argument('--n_reembedding_steps', type=int, default=None, help="Number of training steps after which to recompute the corpus embeddings")

    parser.add_argument('--n_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--lr_type', type=str, default="cosine", help="Learning rate scheduler approach")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Fraction of total training steps for warmup (0.0 to 1.0)")
    parser.add_argument('--train_batch_size', type=int,default=2, help="Batch size for training")
    parser.add_argument('--eval_batch_size', type=int, default=2, help="Batch size for evaluation")
    parser.add_argument('--preprocessing_batch_size', type=int, default=4, help="Batch size for the preprocessing phase")
    parser.add_argument('--padding_side', type=str, default="right", help="Padding side for tokenizers")

    parser.add_argument('--lambda_loss', type=float, default=0.2, help="Lambda weighting factor parameter")
    
    parser.add_argument('--n_neg_examples', type=int, default=3, help="Number of negative samples to include in the triplets")
    parser.add_argument("--k_eval_values_accuracy", nargs="+", type=int, default=[1, 3, 5],
                        help="Values of k for accuracy@k evaluation metrics")
    parser.add_argument("--k_eval_values_ndcg", nargs="+", type=int, default=[1, 3, 5],
                        help="Values of k for ndcg@k evaluation metrics")
    parser.add_argument('--gamma', type=float, default=1, help="Gamma parameter for computing Pr_retr")
    parser.add_argument('--beta', type=float, default=1, help="Beta parameter for softmax in Q computation")
    parser.add_argument('--preference_weight', type=float, default=0.1, help="Weighting factor for the preference ratio")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    parser.add_argument('--wandb_project_name', type=str, default="PortsAAAI", help="WandbB project name")
    parser.add_argument('--wandb_run_name', type=str, default="test_run", help="WandbB run name")

    parser.add_argument('--log_freq', type=int, default=100, help="Logging frequency")

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading Retrieval Model")
    retr_model_name = args.retrieval_model_name
    retr_tokenizer = AutoTokenizer.from_pretrained(retr_model_name)
    retr_model = AutoModel.from_pretrained(retr_model_name).to(device)
    

    logger.info("Loading Generative Model")
    pseudo_model_name = args.inference_model_name
    infer_model_name = pseudo_name_mapping[pseudo_model_name]
    infer_tokenizer = AutoTokenizer.from_pretrained(infer_model_name)
    if args.load_in_4bit:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        infer_model = AutoModelForCausalLM.from_pretrained(infer_model_name, 
                                                           quantization_config=nf4_config, 
                                                           device_map=0 if device=="cuda" else "cpu",
                                                           attn_implementation="flash_attention_2")
    else:
        infer_model = AutoModelForCausalLM.from_pretrained(infer_model_name, 
                                                           device_map=0 if device=="cuda" else "cpu",
                                                           attn_implementation="flash_attention_2")

    
    if infer_tokenizer.pad_token is None:
        logger.info("No padding token - using EOS instead")
        infer_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
        infer_tokenizer.pad_token_id = infer_tokenizer.eos_token_id

        infer_model.resize_token_embeddings(len(infer_tokenizer))

    infer_tokenizer.padding_side=args.padding_side

    if retr_tokenizer.pad_token is None:
        logger.info("No padding token - using EOS instead")
        retr_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        retr_tokenizer.pad_token = retr_tokenizer.eos_token
        retr_tokenizer.pad_token_id = retr_tokenizer.eos_token_id

        retr_model.resize_token_embeddings(len(retr_tokenizer))

    retr_tokenizer.padding_side=args.padding_side

    logger.info("Loading dataset")
    dataset_downloader = DatasetDownloader(dataset_name=args.dataset)
    dataset = dataset_downloader.get_dataset()

    if args.dataset in ["apibench", "toolbench"]:
        global query_id_dict, apis_multi
        query_id_dict = {x : i for i,x in enumerate(list(set(dataset["test"]["query_for_retrieval"])))}
        apis_multi = {i : list({y["api_description"] for y in dataset["test"].filter(lambda z : z["query_for_retrieval"] == x)}) for x, i in query_id_dict.items()}

    dataset = dataset_downloader.post_process_answers(dataset)

    if args.max_train_samples:
        n_inst = min(args.max_train_samples, len(dataset["train"]))
        selected_indices = random.sample(range(len(dataset["train"])), n_inst)
        dataset["train"] = dataset["train"].select(selected_indices)

    if args.max_eval_samples:
        n_inst = min(args.max_eval_samples, len(dataset["test"]))
        selected_indices = random.sample(range(len(dataset["test"])), n_inst)
        dataset["test"] = dataset["test"].select(selected_indices)

    logger.info(">>> DATASET STATS")
    for split_name in dataset:
        logger.info(f"  {split_name}: {len(dataset[split_name])} examples")

    logger.info("Defining tool corpora")
    
    train_api_corpus = list(set(dataset["train"]["api_description"]))
    eval_api_corpus = list(set(dataset["test"]["api_description"]))

    logger.info(f"Corpus sizes: Train: {len(train_api_corpus)} | Eval: {len(eval_api_corpus)}")

    logger.info("Setting the prompt and answer templates")
    prompt_template = PROMPT_TEMPLATES[pseudo_model_name]
    instruction = pseudo_name_instr_mapping[pseudo_model_name]
    answer_template = prompt_template["answer_template"]

    response_template_ids = infer_tokenizer.encode(answer_template,
                                                    add_special_tokens=False)

    data_collator_completion = DataCollatorForCompletionOnlyLM(tokenizer=infer_tokenizer,
                                                                response_template=response_template_ids,
                                                                mlm=False)

    train_eval_config = {
        "dataset" : dataset,
        "dataset_name" : args.dataset,
        "retr_tokenizer" : retr_tokenizer, 
        "retr_model" : retr_model,
        "infer_tokenizer" : infer_tokenizer,
        "infer_model" : infer_model,
        "train_api_corpus" : train_api_corpus,
        "eval_api_corpus" : eval_api_corpus,
        "eval_strategy" : args.eval_strategy,
        "eval_steps" : args.eval_steps,
        "n_reembedding_steps" : args.n_reembedding_steps,
        "prompt_template" : prompt_template["prompt_template"],
        "instruction_prompt" : instruction,
        "data_collator_completion" : data_collator_completion,
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
        "log_freq" :  args.log_freq,
        "k_eval_values_accuracy" : args.k_eval_values_accuracy,
        "k_eval_values_ndcg" : args.k_eval_values_ndcg,
        "device" : device,
        "wandb_project_name" : args.wandb_project_name,
        "wandb_run_name" : args.wandb_run_name,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_dir": args.save_dir,
        "max_checkpoints": args.max_checkpoints
    }
    logger.info("Starting Training and Evaluation")
    train(**train_eval_config)
    logger.info("Success. Exit.")


if __name__ == "__main__":
    main()