import os
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    get_scheduler, 
    HfArgumentParser,
    set_seed
)
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from sklearn.metrics import ndcg_score
from huggingface_hub import login
from src.data_classes import PyTorchTrainingParams
from src.prompts import PROMPT_TEMPLATES, INSTRUCTIONS
import logging
import wandb

def main():
    # parse arguments and log to cloud services
    parser = HfArgumentParser(PyTorchTrainingParams)
    (args,) = parser.parse_args_into_dataclasses()
    log_wandb = args.log_to_wandb
    verbose = args.verbose
    hf_key = os.getenv('HF_KEY')
    login(token=hf_key)
    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    def log_mem_usage(topic):
        current_device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_device)
        memory_reserved = torch.cuda.memory_reserved(current_device)
        logger.info(f"{topic} A: {memory_allocated / 1024**2} MB, R: {memory_reserved / 1024**2} MB")

    
    # load models and tokenizers
    set_seed(args.seed)
    retr_model_name = args.retr_model_name_or_path
    infer_model_name = args.infer_model_name_or_path
    retr_tokenizer = AutoTokenizer.from_pretrained(retr_model_name)
    retr_model = AutoModel.from_pretrained(retr_model_name).to("cuda")
    infer_tokenizer = AutoTokenizer.from_pretrained(infer_model_name, trust_remote_code=True)
    if infer_tokenizer.pad_token is None:
        logger.info("No padding token - using EOS instead")
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
    infer_tokenizer.padding_side='left'
    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False if args.quantization_4bit else True,
            load_in_4bit=True if args.quantization_4bit else False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        infer_model = AutoModelForCausalLM.from_pretrained(
            infer_model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
    else:
        infer_model = AutoModelForCausalLM.from_pretrained(
            infer_model_name, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).to("cuda")
    if verbose:
        logger.info(f"Model dtype: {next(infer_model.parameters()).dtype}")

    # initialize wandb run
    if log_wandb:
        wandb_key = os.getenv('WANDB_KEY')
        name = args.wandb_proj_name
        wandb.login(key=wandb_key)
        wandb.init(
            project='fine-tuning-retriever',
            name=name if name else f"training run - {args.dataset_path.split('/')[-1]}",
            config={
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
                "dataset_path": args.dataset_path,
            }
        )
        wandb.watch(retr_model, log_freq=10)

    retr_model.eval()
    infer_model.eval()

    def compute_embeddings(corpus, batch_size=16):
        all_embeddings = []
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i+batch_size]
            batch = retr_tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            batch = {k:batch[k].to("cuda") for k in batch}
            out = retr_model(**batch) 
            embeddings = F.normalize(out[0][:, 0], p=2, dim=1)
            if len(embeddings.shape) > 2:
                embeddings = embeddings.squeeze(0)
            all_embeddings.append(embeddings)
        stacked_embeddings = torch.cat(all_embeddings, dim=0)
        return stacked_embeddings

    # data preparation
    infer_model_type = args.infer_model_type
    INSTRUCTION = INSTRUCTIONS["function_calling_groq" if infer_model_type == 'llama3groq' else "function_calling"]
    prompt_template = PROMPT_TEMPLATES[infer_model_type]["prompt_template"]
    ANSWER = PROMPT_TEMPLATES[infer_model_type]["answer_template"]
    
    dataset = load_dataset(args.dataset_path, "parsed_data")
    if args.dataset_path == "ToolRetriever/ToolBench":
        dataset['train'] = dataset['train'].filter(lambda x: x['group'] == 'G3' and bool(x['api_description'].strip()))
        dataset = dataset['train'].train_test_split(test_size=0.3, seed=42)
    elif args.dataset_path == "ToolRetriever/BFCL":
        dataset = dataset['test'].train_test_split(test_size=0.3, seed=42)
    elif args.dataset_path == "ToolRetriever/APIBench":
        ds_dict = {}
        for split in dataset:
            for row in dataset[split]:
                answ = row["answer"]
                descr = row["api_description"]

                if answ not in ds_dict: ds_dict[answ] = descr

        def unique_descr(example):
            out = example
            out["api_description"] = ds_dict[example["answer"]]
            return out

        dataset = dataset.map(unique_descr)

    dataset = dataset.shuffle(seed=42).flatten_indices()

    query_column = args.query_column
    response_column = args.response_column
    train_documents = list(set(dataset["train"]["api_description"]))
    eval_documents = list(set(dataset["test"]["api_description"]))

    def tokenize_function(samples):
        return retr_tokenizer(samples[query_column], padding=True, truncation=True, return_tensors='pt')

    input_training_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    input_eval_dataset = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["test"].column_names
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

    optimizer = AdamW(retr_model.parameters(), lr=args.learning_rate)
    num_training_steps = num_epochs * len(train_data_loader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    cross_entropy = CrossEntropyLoss(reduction='none')
    kl_div = KLDivLoss(reduction='none')

    # inner data utilities
    inner_data_collator = DataCollatorForCompletionOnlyLM(
        response_template=ANSWER, 
        tokenizer=infer_tokenizer, 
        mlm=False)

    def inner_tokenize_function(samples):
        return infer_tokenizer(samples["text"], truncation=True, padding=True, return_tensors="pt")
    
    # training utility functions
    def parse_batch(dataset_split, batch, index):
        """
        From the batch tokenized by the retriever's tokenizer, get the queries and responses in string format
        """
        curr_bs = batch["input_ids"].size(0)
        batch_data = dataset_split[index*batch_size:(index*batch_size)+curr_bs]
        return curr_bs, batch_data


    def get_top_k_docs_per_query(embedded_documents, batch, k):
        """
        Compute the top-k documents for each query in the batch, based on their cosine similarity
        """
        embedded_queries = compute_embeddings(batch[query_column])
        embedded_documents_exp = embedded_documents.unsqueeze(0)  # Size becomes [1, n_docs, embeddings_size]
        embedded_queries_exp = embedded_queries.unsqueeze(1)  # Size becomes [batch_size, 1, embeddings_size]
        cos_sim = F.cosine_similarity(embedded_documents_exp, embedded_queries_exp, dim=-1)  # Size becomes [batch_size, n_docs]
        top_k_docs = torch.topk(cos_sim, k, dim=-1)  # Size becomes [batch_size, k]
        return top_k_docs.indices, top_k_docs.values, cos_sim

    compute_Pr = lambda similarities, gamma: F.softmax(similarities / gamma, dim=1)

    def get_prompts(prompt_template, documents, batch_data, documents_per_query):
        """
        Makes prompts to pass to the inference model using the respective documents for each query
        """
        prompts = [
                prompt_template.format(
                    INSTRUCTION,
                    documents[doc_index], 
                    batch_data[query_column][data_index], 
                    batch_data[response_column][data_index]
                )
                for i_th_doc in range(documents_per_query.size(1))
                for data_index, doc_index in enumerate(documents_per_query[:, i_th_doc])
            ]
        return prompts

    def prepare_inner_data_loader(prompts, curr_bs, inner_tokenize_function, inner_data_collator):
        """
        From the prompts, prepares the data to pass to the inference model
        Makes k batches of size curr_bs
        """
        inner_dataset = Dataset.from_pandas(pd.DataFrame(prompts, columns=["text"]))
        inner_dataset = inner_dataset.map(
            inner_tokenize_function,
            batched=True,
            remove_columns=inner_dataset.column_names
        )
        inner_data_loader = DataLoader(
            inner_dataset, shuffle=False, batch_size=curr_bs, collate_fn=inner_data_collator
        )
        return inner_data_loader

    def get_perplexity_per_sample(outputs, labels, cross_entropy):
        """
        From the inference model's outputs and the labels, compute the perplexity of each example
        """
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

    def compute_Q(perplexities, beta):
        perplexities = torch.stack(perplexities).T
        Q = F.softmax(perplexities / beta, dim=1)
        return Q

    def compute_loss(Q, Pr, kl_div):
        """
        Computes KL(Pr||Q)
        """
        Q_log = torch.log(Q)
        divergence = kl_div(Q_log, Pr).sum(-1)
        loss = divergence.mean()
        return loss

    # evaluation utility functions
    verify_relevancy = lambda api, doc: api == doc

    def get_relevant_docs(batch_data, documents):
        """
        For each example in the batch, retrieve its relevant document
        """        
        rel_docs = []
        for api in batch_data["api_description"]:
            for i, doc in enumerate(documents):
                if verify_relevancy(api, doc):
                    rel_docs.append(i)
                    break
        return torch.tensor(rel_docs).unsqueeze(-1)

    # CHANGED FUNCTIONS
    ############################################################################################################
    
    def verify_relevancy_new(api, doc):
        return api == doc if isinstance(api, str) else api in doc

    def get_relevant_docs_new(batch_data, documents):
        """
        For each example in the batch, retrieve its relevant document
        """        
        rel_docs = []
        for api in batch_data["api_description"]:
            cur_rel = []
            for i, doc in enumerate(documents):
                if verify_relevancy_new(api, doc):
                    cur_rel.append(i)
            rel_docs.append(torch.tensor(cur_rel).to("cuda"))
        return rel_docs

    ############################################################################################################
    
    def accumulate_ranks_at_n(ranks, documents_per_query, documents, batch_data, k):
        """
        Compute the ranks@n for the batch, sum them up to the previous batches' values
        """
        rel = get_relevant_docs(batch_data, documents).to("cuda")
        ### TEST
        rel_new = get_relevant_docs_new(batch_data, documents)
        ###
        if verbose:
            logger.info(f"DOCS PER QUERY\n{documents_per_query}")
            logger.info(f"RELEVANT DOCS\n{rel}")
        for n in range(k):
            rank_at_n = torch.any(documents_per_query[:, :n+1] == rel, dim=-1).sum()
            ### TEST
            rank_at_n_new = 0
            for i, ex_docs in enumerate(rel_new):
                rank_at_n_new += torch.isin(documents_per_query[i, :n+1], ex_docs).sum() / ex_docs.size(0)
            assert torch.allclose(rank_at_n.float(), rank_at_n_new, atol=1e-4), "rank_at_n computation is not equal"
            ###
            ranks[n] += rank_at_n.item()
        return ranks

    def accumulate_ndcg_scores(ndcg_scores, ndcg_k_values, batch_data, curr_bs, cos_sim):
        """
        Compute the NDCG scores for the batch
        """
        for ex_index in range(curr_bs):
            true_relevance = np.array([
                verify_relevancy(batch_data["api_description"][ex_index], doc)
                for doc in (eval_documents)
            ])
            ### TEST
            true_relevance_new = np.array([
                verify_relevancy_new(batch_data["api_description"][ex_index], doc)
                for doc in (eval_documents)
            ])
            assert np.allclose(true_relevance, true_relevance_new, atol=1e-4), "true_relevance computation is not equal"
            ###
            for i, k_val in enumerate(ndcg_k_values):
                ndcg_scores[i].append(
                    ndcg_score([true_relevance], [cos_sim[ex_index].cpu().numpy()], k=k_val)
                )

    # evaluation function
    def evaluate(embedded_documents):
        with torch.no_grad():
            ranks = [0 for _ in range(k)]
            ndcg_k_values = [1, 3, 5]
            ndcg_scores = [[] for _ in range(len(ndcg_k_values))]
            for index, batch in enumerate(eval_data_loader):
                curr_bs, batch_data = parse_batch(dataset["test"], batch, index)
                documents_per_query, similarities_per_query, cos_sim = get_top_k_docs_per_query(embedded_documents, batch_data, k)
                accumulate_ranks_at_n(ranks, documents_per_query, eval_documents, batch_data, k)
                accumulate_ndcg_scores(ndcg_scores, ndcg_k_values, batch_data, curr_bs, cos_sim)
                torch.cuda.empty_cache()
            ranks = [r / len(eval_data_loader.dataset) * 100 for r in ranks]
            rank_avg = sum(ranks[:3]) / 3
            for n in range(k):
                if verbose:
                    logger.info(f"RANK@{n+1}: {ranks[n]}")
                if log_wandb:
                    wandb.log({f"Rank@{n+1}": ranks[n]})
            if verbose:
                logger.info(f"RANKAVG: {rank_avg}")
            if log_wandb:
                wandb.log({"RankAvg": rank_avg})  

            for i, k_val in enumerate(ndcg_k_values):
                ndcg_score_avg = sum(ndcg_scores[i]) / len(eval_data_loader.dataset)
                if verbose:
                    logger.info(f"NDCG@{k_val}: {ndcg_score_avg}")
                if log_wandb:
                    wandb.log({f"NDCG@{k_val}": ndcg_score_avg})

    with torch.no_grad():
        embedded_eval_documents = (compute_embeddings(eval_documents))
    evaluate(embedded_eval_documents)
    del embedded_eval_documents

    retr_model.train()
    for epoch in range(num_epochs):
        for index, batch in enumerate(train_data_loader):
            if verbose:
                logger.info(f"Epoch: {epoch}, Batch: {index}")
            curr_bs, batch_data = parse_batch(dataset["train"], batch, index)
            retr_model.eval()
            with torch.no_grad():
                embedded_documents = compute_embeddings(train_documents)
            retr_model.train()
            log_mem_usage("MEM USAGE")
            # documents_per_query contains the indices of the top-k documents for each query in the batch
            # similarities_per_query contains the cosine similarities of the top-k documents for each query in the batch
            documents_per_query, similarities_per_query, _ = get_top_k_docs_per_query(embedded_documents, batch_data, k)
            Pr = compute_Pr(similarities_per_query, gamma)
            del similarities_per_query, _
            torch.cuda.empty_cache()
            prompts = get_prompts(prompt_template, train_documents, batch_data, documents_per_query)
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
            del perplexities, Q, Pr, inner_data_loader, inner_batch
            torch.cuda.empty_cache()
            if log_wandb:
                wandb.log({"Training Loss": loss.item()})
        retr_model.eval()
        with torch.no_grad():
            embedded_documents = compute_embeddings(eval_documents)
        evaluate(embedded_documents)
        retr_model.train()
        del embedded_documents
        torch.cuda.empty_cache()

    if verbose:
        logger.info("TRAINING FINISHED.")
    if log_wandb:
        wandb.finish()
    if args.trained_model_save_path:
        retr_model.save_pretrained(args.trained_model_save_path)

if __name__ == "__main__":
    main()

