import random
import json

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from typing import List
from tqdm import tqdm
from sklearn.metrics import ndcg_score

import numpy as np

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



# --------------------------------------------------------
# >> TRAINING LOOP HELPERs

def compute_embeddings(model, 
                      documents,
                      device : str = "cuda"):
    """
    Compute the embedding of input documents using the given encoder model.
    """
    # Compute token embeddings
    documents = {k: v.to(device) for k, v in documents.items()}
    model_output = model(**documents)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings_normal = F.normalize(sentence_embeddings, p=2, dim=-1)
    del documents, model_output, sentence_embeddings
    return sentence_embeddings_normal


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

# def compute_similarity(model, 
#                        queries, 
#                        documents,
#                        device : str = "cuda"):
#     """
#     Compute the similarity between queries and documents using the input encoder model.
#     """       
#     # Compute embedding
#     documents_tok = {
#         k : documents[k].to(device) for k in documents
#     }

#     queries_tok = {
#         k : queries[k].to(device) for k in queries
#     }

#     # CLS pooling + normalization
#     docs_embeddings = F.normalize(model(**documents_tok)[0][:, 0], p=2, dim=-1).unsqueeze(0)
#     q_embeddings = F.normalize(model(**queries_tok)[0][:, 0], p=2, dim=-1).unsqueeze(0)

#     del documents_tok, queries_tok

#     cos_sim = F.cosine_similarity(docs_embeddings, q_embeddings, dim=-1)

#     return cos_sim

def compute_similarity(model, queries, documents, device="cuda"):
    """
    Compute the similarity between corresponding queries and documents in a batch.
    """       
    # Move to device
    documents_tok = {
        k : documents[k].to(device) for k in documents
    }

    queries_tok = {
        k : queries[k].to(device) for k in queries
    }
    
    # CLS pooling + normalization (without unsqueeze)
    docs_embeddings = F.normalize(model(**documents_tok)[0][:, 0], p=2, dim=-1)
    q_embeddings = F.normalize(model(**queries_tok)[0][:, 0], p=2, dim=-1)
    
    # Compute batch-wise cosine similarity (corresponding pairs)
    cos_sim = torch.sum(docs_embeddings * q_embeddings, dim=1, keepdim=True)
    
    return cos_sim


def get_gradient_norm(model):
    """
    Get the current norm of gradients for the input model.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def encode_query(retr_model, 
                 query,
                 device : str = "cuda"):
    """
    Encode the input query with the given encoder model.
    """
    query = {k:query[k].to(device) for k in query}
    enc = retr_model(**query)[0]
    #print(f"ENC SHAPE: {enc.shape}")

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
            
            # batch = retr_tokenizer(batch, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            # batch = {k:batch[k].to(device) for k in batch}
            # out = retr_model(**batch)
            # embeddings = F.normalize(out[0][:, 0], p=2, dim=-1)

            embeddings = retr_model.encode(batch, 
                                           show_progress_bar=False,
                                           convert_to_tensor=True,
                                           device=device)
            if len(embeddings.shape) > 2:
                embeddings = embeddings.squeeze(0)
            
            all_embeddings.append(embeddings.cpu())
            

    return torch.cat(all_embeddings, dim=0)







# def old_train(dataset : Dataset,
#           retr_tokenizer : AutoTokenizer, 
#           retr_model : AutoModel,
#           infer_tokenizer : AutoTokenizer,
#           infer_model : AutoModelForCausalLM,
#           train_api_corpus : List[str],
#           eval_api_corpus : List[str],
#           lambda_loss : float = 0.2,
#           train_batch_size : int = 2,
#           eval_batch_size : int = 2,
#           num_epochs : int = 10,
#           retriever_max_seq_length : int = 514,
#           inference_max_seq_length : int = 1024,
#           number_of_neg_examples : int = 3,
#           log_freq :  int = 100,
#           k_eval : int = 10,
#           device : str = "cuda"):
    
#     # Initialize wandb
#     wandb.init(project=PROJECT_NAME, 
#                name=RUN_NAME)


#     retr_model.train()
#     for epoch in range(num_epochs):

#         logger.info("Creating pseudo-random dataloader")
#         train_data_config = {
#             "dataset" : dataset,
#             "api_corpus_list" : train_api_corpus,
#             "retrieval_max_length" : retriever_max_seq_length,
#             "generateor_max_length" : inference_max_seq_length,
#             "retrieval_tokenizer" : retr_tokenizer,
#             "inference_tokenizer" : infer_tokenizer,
#             "epoch_number" : epoch,
#             "batch_size" : train_batch_size,
#             "num_neg_examples" : number_of_neg_examples
#         }
#         triplet_dataloader = get_train_dataloader(**train_data_config)

#         logger.info(f"Starting training epoch {epoch+1}/{num_epochs}")

#         pbar = tqdm(enumerate(triplet_dataloader), total=len(triplet_dataloader), desc="Training PORT with RePlug+ORPO")
#         for bid, batch in pbar:

#             bs = batch["query"]["input_ids"].shape[0]
#             n_neg_docs = neg_docs[0]["input_ids"].shape[0]

#             pos_labels = batch["labels_pos"]    # [bs, max_seq_len]
#             neg_labels = batch["labels_neg"]    # bs * [n_neg_docs, max_seq_len]

#             queries = batch["query"]            # [bs, max_seq_len]
#             pos_docs = batch["positive"]        # [bs, max_seq_len]
#             neg_docs =  batch["negative"]       # bs * [n_neg_docs, max_seq_len]


#             # > Compute positive-negative similarities
#             pos_simialrity = compute_similarity(retr_model, queries, pos_docs).view(bs,-1)  # [bs,1]


#             neg_similarity = []
#             for nid in range(n_neg_docs):
#                 # consider the i-th negative document from each batch
#                 data = {
#                     k : torch.stack([neg_docs[bid][k][nid,:] for bid in range(bs)]) for k in ['input_ids', 'attention_mask']
#                 }
#                 this_neg_similarity = compute_similarity(retr_model, 
#                                                         queries, 
#                                                         data).view(bs,-1)
#                 neg_similarity.append(this_neg_similarity)
            

#             neg_similarity = torch.stack(neg_similarity, dim=-1)
#             similarities = torch.cat((pos_simialrity, neg_similarity.squeeze(1)), dim=-1) # [bs, 1+n_neg_docs]


#             # Normalize and weight similarities into retrieval probabilities
#             Pr_retr = compute_Pr(
#                 similarities = similarities,
#                 gamma = gamma,
#                 axis = -1
#             )

#             # > Get prompts
#             input_prompt_pos = batch["q_pos_prompt"]
#             input_prompt_neg = batch["q_neg_prompt"]

#             # > Inference on the model
#             input_prompt_pos = {k : input_prompt_pos[k].to(device) for k in input_prompt_pos}
#             input_prompt_neg = [{k : neg_docs_trip[k].to(device) for k in neg_docs_trip} for neg_docs_trip in input_prompt_neg]

#             pos_labels = {k : pos_labels[k].to(device) for k in pos_labels}
#             neg_labels = [{k : neg_docs_trip[k].to(device) for k in neg_docs_trip} for neg_docs_trip in neg_labels]

#             # > Define dataloader with response_template masking
#             pos_inner_data_loader = DataLoader(
#                 Dataset.from_dict(input_prompt_pos),
#                 shuffle=False,
#                 batch_size=bs,
#                 collate_fn=data_collator_completion
#             )

#             neg_inner_data_loader = [DataLoader(
#                 Dataset.from_dict({k : torch.stack([input_prompt_neg[bid][k][nid,:] for bid in range(bs)]) for k in ["input_ids", "attention_mask"]}),
#                 shuffle=False,
#                 batch_size=bs,
#                 collate_fn=data_collator_completion
#             ) for nid in range(n_neg_docs)]

#             pos_perplexity = []
#             neg_perplexity = []

#             with torch.no_grad():

#                 # >> RUN FORWARD

#                 # > POSITIVE
#                 # - move data to device
#                 _,pos_data = next(enumerate(pos_inner_data_loader))
#                 pos_data = {k:pos_data[k].to(device) for k in pos_data}
#                 pos_data["labels"] = pos_labels["input_ids"].to(device)

#                 # - forward
#                 outputs_pos = infer_model(**pos_data)

#                 # - compute perplexity
#                 pos_ppl_config = {
#                     #"loss" : outputs_pos.loss,
#                     "outputs" : outputs_pos,
#                     "input_ids" : pos_data["input_ids"],
#                     "attention_mask" : pos_data["attention_mask"]
#                 }
#                 pos_perplexity = get_perplexity(**pos_ppl_config)


#                 # >> COMPUTE LOG-PROBABILITY FOR ORPO
#                 pos_logps = get_batch_logps(logits=outputs_pos.logits,
#                                             labels=pos_data["labels"], # masked with response template
#                                             average_log_prob=True,
#                                             label_pad_token_id=infer_tokenizer.pad_token_id)

#                 del outputs_pos
#                 torch.cuda.empty_cache()

#                 # > NEGATIVES
#                 neg_data_out = []
#                 neg_logps = []

#                 for n_id in range(n_neg_docs):

#                     # - compose data and move to device
#                     _,neg_data = next(enumerate(neg_inner_data_loader[n_id]))
#                     neg_data = {k:neg_data[k].to(device) for k in neg_data}
#                     _neg_labels = torch.stack([neg_labels[bid]["input_ids"][n_id,:] for bid in range(bs)], dim=0)
#                     neg_data["labels"] = _neg_labels.to(device)

#                     # - forward
#                     outputs_neg = infer_model(**neg_data)
#                     neg_data_out.append(outputs_neg)

#                     # - compute perplexity
#                     neg_ppl_config = {
#                         "outputs" : outputs_neg,
#                         "input_ids" : neg_data["input_ids"],
#                         "attention_mask" : neg_data["attention_mask"]
#                     }
#                     _neg_perplexity = get_perplexity(**neg_ppl_config)
#                     neg_perplexity.append(_neg_perplexity)

#                     # >> COMPUTE LOG-PROBABILITY FOR ORPO
#                     this_neg_logps = get_batch_logps(logits=outputs_neg.logits,
#                                                     labels=neg_data["labels"],
#                                                     average_log_prob=True,
#                                                     label_pad_token_id=infer_tokenizer.pad_token_id)
#                     neg_logps.append(this_neg_logps)

#                     del outputs_neg
#                     torch.cuda.empty_cache()

#             del pos_inner_data_loader, neg_inner_data_loader
#             torch.cuda.empty_cache()


#             # >> Compute RePLUG Loss
#             neg_perplexity = torch.stack(neg_perplexity, dim=-1)
#             concat_perplexities = torch.cat((pos_perplexity.unsqueeze(0).T, neg_perplexity), dim=-1)


#             Q = F.softmax(concat_perplexities / beta, dim=-1) # [bs,1+n_docs]

#             del neg_perplexity, pos_perplexity
#             torch.cuda.empty_cache()

#             replug_loss = compute_loss_replug(Q, Pr_retr, kl_div)

#             # >> Compute ORPO Loss
#             orpo_loss, pref_ratio, log_odds_chosen = 0,0,0
#             pos_rewards, neg_rewards = [],[]

#             # - for each pair (pos, neg_i)
#             for neg_i_logps in neg_logps:
#                 _orpo_loss, _pos_reward, _neg_reward, _pref_ratio, _log_odds_chosen = odds_ratio_loss(
#                     positive_retr_log_prob=pos_logps,
#                     negative_retr_log_prob=neg_i_logps,
#                     beta=0.1
#                 )
#                 orpo_loss += _orpo_loss.mean()
#                 pos_rewards.append(_pos_reward.mean())
#                 neg_rewards.append(_neg_reward.mean())
#                 pref_ratio += _pref_ratio
#                 log_odds_chosen += _log_odds_chosen


#             orpo_loss = orpo_loss.mean()
#             del neg_logps, pos_logps
#             torch.cuda.empty_cache()

#             loss = replug_loss - lambda_loss * orpo_loss

#             if math.isinf(loss):
#                 logger.info(f"Discarded infinite loss perturbation")
#             else:
#                 loss.backward()

#             # Compute gradient norm
#             grad_norm = get_gradient_norm(retr_model)

#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()

#             reward_accuracies = (torch.tensor(pos_rewards) > torch.tensor(neg_rewards)).float()
#             pbar.set_postfix({'Loss': loss.item(), 'RePlug Loss' : replug_loss.item(), "OPRO Loss" : -orpo_loss.item(), "Reward Accuracy" : reward_accuracies.mean().cpu().item()})


#             if bid % log_freq == 0 or bid == 0:
                
#                 wandb.log({
#                     "port_loss": loss.item(),
#                     "replug_loss": replug_loss.item(),
#                     "odds_ratio_loss": -orpo_loss.item(),
#                     "ratio_reward" : pref_ratio,
#                     "reward_Accuracy" : reward_accuracies.mean().cpu(),
#                     "positive_reward" : pos_reward.detach().mean().cpu(),
#                     "negative_reward" : neg_reward.detach().mean().cpu(),
#                     "positive_odds_chosen" : log_odds_chosen,
#                     "positive_logpb" : pos_logps.detach().mean().cpu(),
#                     "negative_logpb" : neg_logps.detach().mean().cpu(),
#                     "gradient_norm" : grad_norm,
#                     "learning_rate": optimizer.param_groups[0]['lr'],
#                     "epoch" : epoch + 1
#                 })

#             del Q, Pr_retr # neg_perplexity
#             torch.cuda.empty_cache()

#         retr_model.eval()
#         eval_corpus_embeddings = embed_corpus(retr_model,
#                                             eval_api_corpus,
#                                             device,
#                                             batch_size=eval_batch_size,
#                                             max_length=retriever_max_seq_length)
#         eval_corpus_embeddings = eval_corpus_embeddings.to(device)

#         ranks = evaluate(retr_model,
#                         eval_triplet_dataloader,
#                         eval_corpus_embeddings,
#                         device,
#                         k=k_eval)

#         # Print results
#         print("\n\n")
#         print("EVALUATION")
#         print("*"*50)
#         for n in range(K_EVAL):
#             print(f"RANK@{n+1}: {ranks[n]:.2f}%")
#         print("*"*50)
#         print("\n\n")

#         wandb.log({
#             f"RANK@{n+1}: {ranks[n]:.2f}%" for n in range(K_EVAL)
#         })

#         retr_model.train()
#         del embedded_documents
#         torch.cuda.empty_cache()

#     wandb.finish()