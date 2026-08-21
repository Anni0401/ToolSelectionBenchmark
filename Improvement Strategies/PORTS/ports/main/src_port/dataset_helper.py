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
            #batch = retr_tokenizer(batch, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            batch = retr_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            batch = {k:batch[k].to(device) for k in batch}

            out = retr_model(**batch)

            embeddings = F.normalize(out[0][:, 0], p=2, dim=-1)
            if len(embeddings.shape) > 2:
                embeddings = embeddings.squeeze(0)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

def compute_embeddings(model, documents, device="cuda"):
    """
    Compute the embedding of input documents using the given encoder model.
    """
    documents = {k: v.to(device) for k, v in documents.items()}
    with torch.no_grad():
        model_output = model(**documents)
    sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings_normal = F.normalize(sentence_embeddings, p=2, dim=-1)

    del sentence_embeddings
    torch.cuda.empty_cache()

    return sentence_embeddings_normal

def create_triplets_with_similar_negatives(dataset, 
                                           retr_model, 
                                           tokenizer, 
                                           num_negatives_per_positive=1, 
                                           retrieval_max_length=512, 
                                           split='train', 
                                           device="cuda", 
                                           preprocessing_batch_size=32):
    triplets = []
    questions = dataset[split]["query_for_retrieval"]
    augmented_descriptions = dataset[split]["api_description"]
    answers = dataset[split]["answer"]

    # Build mapping from queries to all indices of their positive API descriptions
    query_to_positive_indices = {}
    for i, query in enumerate(questions):
        if query not in query_to_positive_indices:
            query_to_positive_indices[query] = []
        query_to_positive_indices[query].append(i)

    batch_size = preprocessing_batch_size

    doc_embeddings = embed_corpus(retr_model, 
                                tokenizer,
                                augmented_descriptions, 
                                device, 
                                batch_size=batch_size, 
                                max_length=retrieval_max_length)
    doc_embeddings = doc_embeddings.to(device)

    # Process queries in batches
    for i in tqdm(range(0, len(questions), batch_size), desc="Processing queries"):
        batch_questions = questions[i:i+batch_size]
        
        # Compute query embeddings for the batch
        query_encodings = tokenizer(batch_questions, truncation=True, max_length=retrieval_max_length, padding='max_length', return_tensors='pt')
        query_embeddings = compute_embeddings(retr_model, query_encodings, device)
        query_embeddings = query_embeddings.to(device)

        # Compute similarities for the batch
        similarities = F.cosine_similarity(query_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2)

        for j, similarity in enumerate(similarities):
            idx = i + j
            query = questions[idx]
            positive = augmented_descriptions[idx]
            
            # Get indices of all positives for this query
            positive_indices = set(query_to_positive_indices[query])
            
            # Sort indices by similarity (descending)
            similar_indices = similarity.argsort(descending=True).tolist()
            
            # Filter out all positive indices
            negative_indices = []
            for sim_idx in similar_indices:
                if sim_idx not in positive_indices:
                    negative_indices.append(sim_idx)
                    if len(negative_indices) == num_negatives_per_positive:
                        break
            
            # Check if we found enough negatives
            if len(negative_indices) < num_negatives_per_positive:
                print(f"Warning: Could not find {num_negatives_per_positive} negatives for query '{query}'. Found {len(negative_indices)}.")

            triplets.append({
                'query': query,
                'positive': positive,
                'negative': [augmented_descriptions[idx] for idx in negative_indices],
                'pos_answer': answers[idx],
                'neg_answer': [answers[idx] for idx in negative_indices]
            })
            
        del query_embeddings
        torch.cuda.empty_cache()

    del similarities, doc_embeddings
    torch.cuda.empty_cache()

    return triplets

# def create_triplets_with_similar_negatives(dataset, 
#                                            retr_model, 
#                                            tokenizer, 
#                                            num_negatives_per_positive=1, 
#                                            retrieval_max_length=512, 
#                                            split='train', 
#                                            device="cuda", 
#                                            preprocessing_batch_size=32):
#     triplets = []
#     questions = dataset[split]["query_for_retrieval"]
#     augmented_descriptions = dataset[split]["api_description"]
#     answers = dataset[split]["answer"]

#     batch_size=preprocessing_batch_size

#     doc_embeddings = embed_corpus(retr_model, 
#                                 tokenizer,
#                                 augmented_descriptions, 
#                                 device, 
#                                 batch_size=batch_size, 
#                                 max_length=retrieval_max_length)
#     doc_embeddings = doc_embeddings.to(device)

#     # Process queries in batches
#     for i in tqdm(range(0, len(questions), batch_size), desc="Processing queries"):
#         batch_questions = questions[i:i+batch_size]
        
#         # Compute query embeddings for the batch
#         query_encodings = tokenizer(batch_questions, truncation=True, max_length=retrieval_max_length, padding='max_length', return_tensors='pt')
#         query_embeddings = compute_embeddings(retr_model, query_encodings, device)
#         query_embeddings = query_embeddings.to(device)

#         # Compute similarities for the batch
#         similarities = F.cosine_similarity(query_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2)

#         for j, similarity in enumerate(similarities):
#             idx = i + j
#             query = questions[idx]
#             positive = augmented_descriptions[idx]

#             # Get top k+1 similar indices
#             top_k = num_negatives_per_positive + 1
#             similar_indices = similarity.argsort(descending=True).tolist() #[:top_k].tolist()

#             # Check if positive is in top k and handle accordingly
#             if idx in similar_indices:
#                 similar_indices.remove(idx)
#                 negative_indices = similar_indices#[:num_negatives_per_positive]
#             else:
#                 negative_indices = similar_indices#[:num_negatives_per_positive]#[idx for idx in similar_indices[:top_k] if idx != idx]

#             negative_indices = negative_indices[:num_negatives_per_positive]

#             assert len(negative_indices) == num_negatives_per_positive

#             triplets.append({
#                 'query': query,
#                 'positive': positive,
#                 'negative': [augmented_descriptions[idx] for idx in negative_indices],
#                 'pos_answer': answers[idx],
#                 'neg_answer': [answers[idx] for idx in negative_indices]
#             })
#         del query_embeddings
#         torch.cuda.empty_cache()

#     del similarities, doc_embeddings
#     torch.cuda.empty_cache()

#     return triplets

# def create_triplets_with_similar_negatives(dataset, 
#                                           retr_model, 
#                                           tokenizer, 
#                                           num_negatives_per_positive=1, 
#                                           retrieval_max_length=512, 
#                                           split='train', device="cuda"):
#     triplets = []
#     questions = dataset[split]["query_for_retrieval"]
#     augmented_descriptions = dataset[split]["api_description"]
#     answers = dataset[split]["answer"]

#     # Pre-compute all document embeddings
#     doc_encodings = tokenizer(augmented_descriptions, truncation=True, max_length=retrieval_max_length, padding='max_length', return_tensors='pt')
#     doc_embeddings = compute_embeddings(retr_model, doc_encodings, device)

#     for i in range(len(questions)):
#         query = questions[i]
#         positive = augmented_descriptions[i]

#         # Compute query embedding
#         query_encoding = tokenizer([query], truncation=True, max_length=retrieval_max_length, padding='max_length', return_tensors='pt')
#         query_embedding = compute_embeddings(retr_model, query_encoding, device)

#         # Compute similarities
#         similarities = F.cosine_similarity(query_embedding, doc_embeddings)
        
#         # Get top k+1 similar indices to ensure we have enough after potential removal of positive
#         top_k = num_negatives_per_positive + 1
#         similar_indices = similarities.argsort(descending=True)[:top_k+1].squeeze().tolist()

#         # Check if positive is in top k and handle accordingly
#         if i in similar_indices[:top_k]:
#             # If positive is in top k, remove it and use the next most similar
#             similar_indices.remove(i)
#             negative_indices = similar_indices[:num_negatives_per_positive]
#         else:
#             # If positive is not in top k, use the top k as negatives
#             negative_indices = [idx for idx in similar_indices[:top_k] if idx != i]

#         assert len(negative_indices) == num_negatives_per_positive

#         triplets.append({
#             'query': query,
#             'positive': positive,
#             'negative': [augmented_descriptions[idx] for idx in negative_indices],
#             'pos_answer': answers[i],
#             'neg_answer': [answers[idx] for idx in negative_indices]
#         })

#     return triplets



class DatasetDownloader():

    def __init__(self,
                 dataset_name : str = "octopus",
                 seed : int = 42):

        self.dataset_name = dataset_name
        self.data_sub_split = "parsed_data" if "toolbench" not in dataset_name else "parsed_data_splitted"
        self.seed = seed
        
        base_ds_path = "ToolRetriever"
        dataset_mapping = {
            "bfcl" : "BFCL",
            "apibank" : "APIBank",
            "apibench" : "APIBench",
            "octopus" : "OctopusNonOverlapping",
            "toole" : "ToolENonOverlapping",
            "toole_90_10" : "ToolENonOverlapping",
            "toole_85_15" : "ToolENonOverlapping",
            "toole_75_25" : "ToolENonOverlapping",
            "toole_70_30" : "ToolENonOverlapping",
            "toole_50_50" : "ToolENonOverlapping",
            "toole_35_65" : "ToolENonOverlapping",
            "toolbench" : "ToolBench",
            "toolbench_1" : "ToolBench",
            "toolbench_2" : "ToolBench",
            "toolbench_3" : "ToolBench",
            "toole-overlap" : "ToolEOverlapping",
            "octopus-overlap" : "OctopusOverlapping"
        }

        data_path = "/".join([base_ds_path, dataset_mapping[dataset_name]])
        self.data_path = data_path

    def get_dataset(self):
        """
        Download and return the dataset
        """
        if "toole" in self.dataset_name:# and "_" in self.dataset_name:
            
            split_name = f"parsed_data_{self.dataset_name.split('_',1)[-1]}" if "_" in self.dataset_name else "parsed_data"
            print(f"Loading {self.data_path} - {split_name}")
            ds = load_dataset(self.data_path, split_name)

            if "90" not in self.dataset_name:
                # use the smaller test set as reference for the ablative comparisons
                ds_test = load_dataset("ToolRetriever/ToolENonOverlapping", "parsed_data_90_10",split="test")
                ds["test"] = ds_test

        else:
            print(f"Loading {self.data_path} - {self.data_sub_split}")
            ds = load_dataset(self.data_path, self.data_sub_split)

        if "toolbench" in self.dataset_name:
            split_group = f"G{self.dataset_name.split('_')[-1]}"
            for split in ds:
                ds[split] = ds[split].filter(lambda x : x["group"] == split_group)
        
        if self.dataset_name == "bfcl":
            unique_k = list(ds.keys())[0]
            ds = ds[unique_k].train_test_split(test_size=0.3, seed=self.seed)

        return ds
    
    def post_process_answers(self,
                             dataset):
        """
        Parse the dataset's labels by adding answer parameters if needed according to they dataset type.
        """
        def __parse_apibank(example):
            out = example
            out["answer"] = f'{out["answer"]}({out["answer_params"]})'
            return out

        if self.dataset_name == "apibank":
            for k in dataset.keys():
                dataset[k] = dataset[k].map(__parse_apibank)

        return dataset


# --------------------------------------------------------
# >> DATA HELPERs

def get_prompted_q_doc(prompt_template : str,
                       document : str,
                       query : str,
                       answer : str,
                       instruction_prompt : str):
    """
    Define prompts to pass to the inference model
    """

    MAX_FUN_CHARS = 500

    prompt_fields = {
        "instruction" : instruction_prompt,
        "api_def" : document[:MAX_FUN_CHARS],
        "query" : query,
        "answer" : answer
    }

    prompts = prompt_template.format(**prompt_fields)

    return prompts


def create_triplets_with_unique_multiple_negatives(dataset : Dataset,
                                                   num_negatives_per_positive : int = 1, 
                                                   split : str = 'train',
                                                   start_seed : int = 0):
    """
    Create triplets of (pseudo-) randomly sampled negatives
    """

    triplets = []
    #questions = dataset[split]["query"]
    questions = dataset[split]["query_for_retrieval"]
    augmented_descriptions = dataset[split]["api_description"]
    answers = dataset[split]["answer"]


    for i in range(len(questions)):
        query = questions[i]
        positive = augmented_descriptions[i]

        # Create a list of possible negatives with their indices
        possible_negatives = list(enumerate(augmented_descriptions))
        possible_negatives = [item for item in possible_negatives if item[0] != i] # exclude self from negs

        seed_i = start_seed + i 
        random.seed(seed_i) # set incremental seed for reproducibility

        # Choose multiple unique negatives
        num_unique_negatives = min(num_negatives_per_positive, len(possible_negatives))
        negative_samples = random.sample(possible_negatives, num_unique_negatives)

        triplets.append({
            'query': query,
            'positive': positive,
            'negative': [n[1] for n in negative_samples],
            'pos_answer': answers[i],
            'neg_answer': [answers[n[0]] for n in negative_samples]
        })

    return triplets

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


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

class TripletCollator(DataCollatorMixin):
    def __init__(self,
                 retrieval_tokenizer,
                 inference_tokenizer,
                 def_corpus=None,
                 prompt_template : str = "",
                 instruction_prompt : str = "",
                 max_length_retrieval : int = 128,
                 max_length_generator : int = 1024):
        self.retr_tokenizer = retrieval_tokenizer
        self.gen_tokenizer = inference_tokenizer
        self.corpus = def_corpus
        self.prompt_template = prompt_template
        self.instruction_prompt = instruction_prompt
        
        self.max_length_retr = max_length_retrieval
        self.max_length_gen = max_length_generator

    def __call__(self, batch):
        prompt_template = self.prompt_template
        queries = [item['query'] for item in batch]
        positives = [item['positive'] for item in batch]
        negatives = [item['negative'] for item in batch]

        pos_answers = [item['pos_answer'] for item in batch]
        neg_answers = [item['neg_answer'] for item in batch]

        # Queries, Pos, Neg docs
        query_encodings = self.retr_tokenizer(queries, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt')
        positive_encodings = self.retr_tokenizer(positives, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt')
        negative_encodings = [self.retr_tokenizer(neg, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt') for neg in negatives]

        # Prompts for inference
        prompted_q_pos = [get_prompted_q_doc(prompt_template=prompt_template, document=item['positive'], query=item['query'], answer=item['pos_answer'], instruction_prompt=self.instruction_prompt) for item in batch]
        prompted_q_pos_encodings = self.gen_tokenizer(prompted_q_pos,
                                                  truncation=True,
                                                  max_length=self.max_length_gen,
                                                  padding='max_length',
                                                  return_tensors='pt')

        prompted_q_neg = []
        for item in batch:
          sub_prompted_neg = []
          for neg_i in range(len(negatives[0])):
            
            #_p_neg = get_prompted_q_doc(prompt_template=prompt_template, document=item['negative'][neg_i], query=item['query'], answer=item['neg_answer'][neg_i], instruction_prompt=self.instruction_prompt)
            _p_neg = get_prompted_q_doc(prompt_template=prompt_template, document=item['negative'][neg_i], query=item['query'], answer=item['pos_answer'], instruction_prompt=self.instruction_prompt)
            
            sub_prompted_neg.append(_p_neg)
          prompted_q_neg.append(sub_prompted_neg)


        prompted_q_neg_encodings = [self.gen_tokenizer(neg_subset,
                                                        truncation=True,
                                                        max_length=self.max_length_gen,
                                                        padding='max_length',
                                                        return_tensors='pt') for neg_subset in prompted_q_neg]

        # Labels
        labels_encodings_pos = self.gen_tokenizer(pos_answers,
                                                  truncation=True,
                                                  max_length=self.max_length_gen,
                                                  padding='max_length',
                                                  return_tensors='pt')
        labels_encodings_neg = [self.gen_tokenizer(neg_a,
                                                  truncation=True,
                                                  max_length=self.max_length_gen,
                                                  padding='max_length',
                                                  return_tensors='pt') for neg_a in neg_answers]

        # Remove token_type_ids
        for encoding in [query_encodings,
                         positive_encodings,
                         prompted_q_pos_encodings,
                         labels_encodings_pos]:
          encoding.pop('token_type_ids', None)

        for encoding in [negative_encodings,
                         prompted_q_neg_encodings,
                         labels_encodings_neg]:
          for el in encoding:
            el.pop('token_type_ids', None)

        # Get gold retrieval_ids wrt corpus
        gold_indices = []
        for pos_doc in positives:
            pos_idx = self.corpus.index(pos_doc)
            gold_indices.append(pos_idx)
        gold_indices = torch.tensor(gold_indices)


        return {
            'query': query_encodings,
            'positive': positive_encodings,
            'negative': negative_encodings,
            'q_pos_prompt' : prompted_q_pos_encodings,
            'q_neg_prompt' : prompted_q_neg_encodings,
            'labels_pos' : labels_encodings_pos,
            'labels_neg' : labels_encodings_neg,
            'gold_retrieval_ids' : gold_indices
        }


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
        # query_encodings = self.retr_tokenizer(queries, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt')
        # positive_encodings = self.retr_tokenizer(positives, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt')
        query_encodings = self.retr_tokenizer(queries, truncation=True, padding=True, return_tensors='pt')
        positive_encodings = self.retr_tokenizer(positives, truncation=True, padding=True, return_tensors='pt')

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


import json

def get_train_dataloader(dataset,
                         api_corpus_list : List[str],
                         retr_model : AutoModel,
                         retrieval_tokenizer : AutoTokenizer,
                         inference_tokenizer : AutoTokenizer,
                         prompt_template : str = "",
                         retrieval_max_length : int = 514,
                         generateor_max_length : int = 1024,
                         batch_size : int = 2,
                         epoch_number : int = 0,
                         num_neg_examples : int = 2,
                         preprocessing_batch_size : int = 8) -> torch.utils.data.DataLoader:
    """
    Create triplets and return train dataloader
    """
    triplets = create_triplets_with_similar_negatives(dataset,
                                                      retr_model,
                                                      retrieval_tokenizer,
                                                      retrieval_max_length=retrieval_max_length, 
                                                      num_negatives_per_positive=num_neg_examples,
                                                      split='train',
                                                      preprocessing_batch_size=preprocessing_batch_size)

    triplet_dataset = TripletDataset(triplets)

    train_collator = TripletCollator(retrieval_tokenizer=retrieval_tokenizer,
                                    inference_tokenizer=inference_tokenizer,
                                    prompt_template=prompt_template,
                                    def_corpus=api_corpus_list,
                                    max_length_retrieval=retrieval_max_length,
                                    max_length_generator=generateor_max_length)

    triplet_dataloader = DataLoader(triplet_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=train_collator,
                                drop_last=False)
                                    
    return triplet_dataloader


def get_eval_dataloader(dataset,
                        api_corpus_list : List[str],
                        retrieval_tokenizer : AutoTokenizer,
                        retrieval_max_length : int = 514,
                        batch_size : int = 2) -> torch.utils.data.DataLoader:

    eval_data = create_instances_wo_negs(dataset, split="test")
    eval_triplet_dataset = TripletDataset(eval_data)
    
    eval_collator = EvalTripletCollator(retrieval_tokenizer=retrieval_tokenizer,
                                        def_corpus=api_corpus_list,
                                        max_length_retrieval=retrieval_max_length)

    eval_dataloader = DataLoader(eval_triplet_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=eval_collator,
                                drop_last=False)
    
    
    return eval_dataloader, eval_triplet_dataset

