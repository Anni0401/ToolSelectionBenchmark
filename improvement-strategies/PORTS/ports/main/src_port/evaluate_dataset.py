import wandb
import math
from tqdm import tqdm
import argparse
import random
import json

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

from tqdm import tqdm


from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    set_seed
)
import torch

import logging
import sys
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import List
import torch
import random
from transformers import AutoModel

from transformers.data.data_collator import DataCollatorMixin
from torch.utils.data import DataLoader, Dataset


from torch.nn.functional import cosine_similarity

from tqdm import tqdm
import torch
import torch.nn.functional as F

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
import random
import json

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from typing import List
from tqdm import tqdm
from sklearn.metrics import ndcg_score

import numpy as np

def get_ndcg_scores(ndcg_k_values, 
                    batch_data, 
                    gold_ids,
                    similarities,
                    api_corpus = None,
                    eval_triplets = None):
    """
    Compute the NDCG scores for the batch, sum them up to the previous batches' values
    """
    len_corpus = similarities.shape[-1]
    ndcg_scores = [[] for _ in range(len(ndcg_k_values))]

    n_data = batch_data["query"]["input_ids"].shape[0]


    for ex_index in range(n_data):

        true_relevance = np.zeros(len_corpus)
        pos_idx = gold_ids[ex_index].cpu().item()
        true_relevance[pos_idx] = 1

        scores = similarities[ex_index].cpu().numpy()

        for k_index, k_val in enumerate(ndcg_k_values):
            ndcg_scores[k_index].append(
                ndcg_score(
                    [true_relevance],
                    [scores],
                    k=k_val # consider only the highest k scores
                )
            )

    return ndcg_scores


def get_ndcg_scores_multi(ndcg_k_values, batch_data, gold_ids_list, similarities, api_corpus=None, eval_triplets=None):
    len_corpus = similarities.shape[-1]
    ndcg_scores = [[] for _ in range(len(ndcg_k_values))]
    n_data = batch_data["query"]["input_ids"].shape[0]

    for ex_index in range(n_data):
        true_relevance = np.zeros(len_corpus)
        for pos_idx in gold_ids_list[ex_index]:
            true_relevance[pos_idx] = 1

        scores = similarities[ex_index].cpu().numpy()

        for k_index, k_val in enumerate(ndcg_k_values):
            ndcg_scores[k_index].append(
                ndcg_score(
                    [true_relevance],
                    [scores],
                    k=min(k_val, len_corpus)  # Ensure k is not larger than corpus size
                )
            )

    return ndcg_scores


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


class EvalTripletCollator(DataCollatorMixin):
    def __init__(self,
                 retrieval_tokenizer : AutoTokenizer,
                 def_corpus : List[str] = None,
                 max_length_retrieval : int = 128):
        self.retr_tokenizer = retrieval_tokenizer
        self.corpus = def_corpus
        self.max_length_retr = max_length_retrieval


    def __call__(self, batch):
        queries = [item['query'] for item in batch]
        positives = [item['positive'] for item in batch]
        pos_answers = [item['pos_answer'] for item in batch]

        # Queries, Pos, Neg docs
        query_encodings = self.retr_tokenizer(queries, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt')
        positive_encodings = self.retr_tokenizer(positives, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt')
        #query_encodings = self.retr_tokenizer(queries, truncation=True, padding=True, return_tensors='pt')
        #positive_encodings = self.retr_tokenizer(positives, truncation=True, padding=True, return_tensors='pt')

        # Remove token_type_ids
        for encoding in [query_encodings,
                         positive_encodings]:
            encoding.pop('token_type_ids', None)

        # Get gold retrieval_ids wrt corpus
        gold_indices = []
        for pos_doc in positives:
            pos_idx = self.corpus.index(pos_doc)
            gold_indices.append(pos_idx)
        
        # for _idx, el in enumerate(gold_indices):
        #     assert self.corpus[el] == positives[_idx]

        gold_indices = torch.tensor(gold_indices)

        return {
            'query': query_encodings,
            'positive': positive_encodings,
            'gold_retrieval_ids' : gold_indices
        }

def create_instances_wo_negs(dataset : Dataset,
                             split : str = 'train'):
    data = []
    #questions = dataset[split]["query"]
    questions = dataset[split]["query_for_retrieval"]
    augmented_descriptions = dataset[split]["api_description"]
    answer = dataset[split]["answer"]

    for i, q in enumerate(questions):
        row = {
            "query" : q,
            "positive" : augmented_descriptions[i],
            "pos_answer" : answer[i]
        }
        data.append(row)

    return data

def encode_query(retr_model, 
                 query,
                 device : str = "cuda"):
    """
    Encode the input query with the given encoder model.
    """
    query = {k:query[k].to(device) for k in query}
    enc = retr_model(**query)[0]


    return F.normalize(enc[:, 0], p=2, dim=-1).unsqueeze(0)


def embed_corpus(retr_model, 
                 retr_tokenizer,
                 corpus, 
                 device, 
                 batch_size=32, 
                 max_length=1024):
    """
    Create embedding matrix for a corpus of documents.
    """
    retr_model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(corpus), batch_size), desc="Embedding corpus"):
            batch = corpus[i:i+batch_size]
            batch = retr_tokenizer(batch, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            #batch = retr_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            batch = {k:batch[k].to(device) for k in batch}

            out = retr_model(**batch)
            #print(f"OUT SHAPE: {out[0].shape}")
            embeddings = F.normalize(out[0][:, 0], p=2, dim=-1)
            #if len(embeddings.shape) > 2:
            #    embeddings = embeddings.squeeze(0)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

def get_eval_dataloader(dataset, api_corpus_list, retrieval_tokenizer, retrieval_max_length=514, batch_size=2):
    eval_data = create_instances_wo_negs(dataset, split="test")
    eval_triplet_dataset = TripletDataset(eval_data)
    
    eval_collator = EvalTripletCollator(retrieval_tokenizer=retrieval_tokenizer,
                                        def_corpus=api_corpus_list,
                                        max_length_retrieval=retrieval_max_length)

    eval_dataloader = DataLoader(eval_triplet_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=eval_collator,
                                 drop_last=False)  # Set drop_last to False
    
    return eval_dataloader, eval_triplet_dataset

def get_ndcg_scores(ndcg_k_values, batch_data, gold_ids, similarities, api_corpus=None, eval_triplets=None):
    len_corpus = similarities.shape[-1]
    ndcg_scores = [[] for _ in range(len(ndcg_k_values))]
    n_data = batch_data["query"]["input_ids"].shape[0]

    for ex_index in range(n_data):
        true_relevance = np.zeros(len_corpus)
        pos_idx = gold_ids[ex_index].cpu().item()
        true_relevance[pos_idx] = 1

        scores = similarities[ex_index].cpu().numpy()

        for k_index, k_val in enumerate(ndcg_k_values):
            ndcg_scores[k_index].append(
                ndcg_score(
                    [true_relevance],
                    [scores],
                    k=min(k_val, len_corpus)  # Ensure k is not larger than corpus size
                )
            )

    return ndcg_scores  # Return a list of lists, not averaging here


def evaluate(retr_model,
             eval_dataloader,
             eval_triplets,
             corpus_embeddings,
             device : str =  "cuda", 
             k : int = 3,
             api_corpus : List[str] = None,
             retr_tokenizer : AutoTokenizer = None,
             dataset_name : str = "toole"):
    retr_model.eval()

    ranks = [0 for _ in range(k)]

    ndcg_k_values = [1, 3, 5]
    ndcg_scores = [[] for _ in range(len(ndcg_k_values))]
    num_samples = len(eval_dataloader.dataset)

    with torch.no_grad():
        for bid, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        
            queries = batch["query"]
            bs = queries["input_ids"].shape[0]


            gold_ids = []

            for _i, i in enumerate(range(bid*eval_dataloader.batch_size, min((bid+1)*eval_dataloader.batch_size, num_samples))):
                if dataset_name not in ["apibench", "toolbench"]:
                    pos = eval_triplets[i]["positive"]
                    idx = api_corpus.index(pos)
                    gold_ids.append(idx)
                else:
                    answ = eval_triplets[i]["pos_answer"]
                    indices = [i for i,doc in enumerate(api_corpus) if answ in doc]
                    gold_ids.append(indices)

            

            # Compute query embeddings
            query_embeddings = encode_query(retr_model, queries, device)

            # Compute similarities with the entire corpus
            embedded_documents_exp = corpus_embeddings.unsqueeze(0)
            embedded_queries_exp = query_embeddings.view(bs,1,-1)
            all_similarities = F.cosine_similarity(embedded_documents_exp,embedded_queries_exp, dim=-1)
            
            # Compute ranks
            top_k_docs= all_similarities.topk(k, dim=-1, largest=True)
            indices, values = top_k_docs.indices, top_k_docs.values


            indices = indices.view(bs,-1)

            if dataset_name not in ["apibench", "toolbench"]:
                gold_ids = torch.tensor(gold_ids).view(1,-1).to(device)

                # Recall
                for _k in range(k):
                    rank_at_n = torch.any(indices[:, :_k+1] == gold_ids.view(bs,-1), dim=-1).sum()
                    ranks[_k] += rank_at_n.item()
                
                # Compute NDCG score
                this_ndcg_scores = get_ndcg_scores(ndcg_k_values, batch, gold_ids.squeeze(0), all_similarities, api_corpus, [eval_triplets[i] for i in range(bid*bs, bid*bs+bs)])
            else:
                # Recall
                for _k in range(k):
                    for b in range(bs):
                        if any(gold_id in indices[b, :_k+1].cpu().numpy().tolist() for gold_id in gold_ids[b]):
                            ranks[_k] += 1
                
                # NDGC
                this_ndcg_scores = get_ndcg_scores_multi(ndcg_k_values, batch, gold_ids, all_similarities, api_corpus, [eval_triplets[i] for i in range(bid*bs, bid*bs+bs)])
            
            for i, _ in enumerate(this_ndcg_scores):
                ndcg_scores[i] += this_ndcg_scores[i]


    # Normalize ranks
    ranks = [round(r / num_samples * 100, 3) for r in ranks]
    
    ndcg_score_avg = [0 for _ in range(len(ndcg_k_values))]
    for i, k_val in enumerate(ndcg_k_values):
        ndcg_score_avg[i] = round(sum(ndcg_scores[i]) / num_samples,3)

    return ranks, ndcg_score_avg




from datasets import load_dataset


def main():
    device="cuda"

    set_seed(242)

    dataset_mapping = {
        "bfcl" : "BFCL",
        "apibank" : "APIBank",
        "apibench" : "APIBench",
        "octopus" : "OctopusNonOverlapping",
        "toole" : "ToolENonOverlapping",
        "toolbench" : "ToolBench",
        "toole-overlap" : "ToolEOverlapping",
        "octopus-overlap" : "OctopusOverlapping"
    }

    dataset_mapping = {
        "octopus" : "OctopusNonOverlapping",
        "toolbench" : "ToolBench",
        "apibench" : "APIBench"
    }

    dataset_mapping = {
        #"octopus-overlap" : "OctopusOverlapping",
        "toole_35_65" : "ToolENonOverlapping",
        "toole_50_50" : "ToolENonOverlapping",
        "toole_70_30" : "ToolENonOverlapping",
        "toole_90_10" : "ToolENonOverlapping",
    }


    #for model_name in ["BAAI/bge-base-en-v1.5"]:#["ToolBench/ToolBench_IR_bert_based_uncased"]:#, "BAAI/bge-base-en-v1.5"]:
    for model_name in ["FacebookAI/roberta-base"]:
        #print("Loading model")
        #model_name = "ToolBench/ToolBench_IR_bert_based_uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name).to(device)

        for _ds_name in dataset_mapping:
            dataset_name = dataset_mapping[_ds_name]

            #for dataset_name in ["OctopusOverlapping"]:
            _dataset_name = f"ToolRetriever/{dataset_name}"
            print(f">> WORKING WITH  {_ds_name}")
        
            if "_" not in _ds_name:
                dataset = load_dataset(_dataset_name, "parsed_data")
            else:
                dataset = load_dataset(_dataset_name, f"parsed_data_{_ds_name.split('_',1)[-1]}")

            print(dataset)


            if _dataset_name == "ToolRetriever/ToolBench":
                GROUP="G3"
                dataset = dataset["train"].filter(lambda x : x["group"] == GROUP and bool(x['api_description'].strip()))
                dataset = dataset.train_test_split(test_size=0.3, seed=42)
            
            if _dataset_name == "ToolRetriever/BFCL":
                dataset = dataset["test"].train_test_split(test_size=0.3, seed=42)
        


            print("Loading corpus")
            eval_api_corpus = list(set(dataset["test"]["api_description"]))


            print(f"LEN CORPUS: {len(eval_api_corpus)}")

            retriever_max_seq_length=512
            eval_batch_size = 4

            print("Get dataloader")
            eval_data_config = {
                "dataset" : dataset, 
                "api_corpus_list" : eval_api_corpus,
                "retrieval_max_length" : retriever_max_seq_length,
                "retrieval_tokenizer" : tokenizer,
                "batch_size" : eval_batch_size
            }
            eval_triplet_dataloader, eval_triplets = get_eval_dataloader(**eval_data_config)

            logger.info("Embedding Tool Corpus")
            eval_corpus_embeddings = embed_corpus(encoder,
                                                tokenizer,
                                                eval_api_corpus,
                                                device,
                                                batch_size=eval_batch_size,
                                                max_length=retriever_max_seq_length)
            eval_corpus_embeddings = eval_corpus_embeddings.to(device)

            print(f"[STATS] {len(eval_triplet_dataloader.dataset)}")

            print("Starting evaluation")
            ranks, ndcg_scores = evaluate(retr_model=encoder,
                                eval_dataloader=eval_triplet_dataloader,
                                eval_triplets=eval_triplets,
                                corpus_embeddings=eval_corpus_embeddings,
                                device = device, 
                                k = 3,
                                api_corpus = eval_api_corpus,
                                retr_tokenizer = tokenizer, 
                                dataset_name=_ds_name)
            
            # Print results

            k_eval=3
            print("\n")
            print(f"********** {_dataset_name} **********")
            print(f"**** {model_name} *****")
            print("EVALUATION")
            print("*"*50)
            print(">>> R@K")
            for n in range(k_eval):
                print(f"RANK@{n+1}: {ranks[n]:.2f}%")
            print("-"*30)
            print(">>> NDCG@K")
            ndcg_k_values = [1, 3, 5]
            for n, _k in enumerate(ndcg_k_values):
                print(f"NDCG@{_k}: {ndcg_scores[n]:.2f}")
            print("*"*50)
            print("\n")


if __name__ == '__main__':
    main()