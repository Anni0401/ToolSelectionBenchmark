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
from huggingface_hub import login
from src.data_classes import PyTorchTrainingParams
from src.prompts import PROMPT_TEMPLATES, INSTRUCTIONS
import logging
import wandb
from sentence_transformers import SentenceTransformer, models

from src_port.retrieval_evaluator import DeviceAwareInformationRetrievalEvaluator
from src_port.utils import embed_corpus


def main():
    parser = HfArgumentParser(PyTorchTrainingParams)
    (args,) = parser.parse_args_into_dataclasses()
    log_wandb = args.log_to_wandb
    verbose = args.verbose
    hf_key = os.getenv('HF_KEY')
    login(token=hf_key)

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

    set_seed(args.seed)
    retr_model_name = args.retr_model_name_or_path
    infer_model_name = args.infer_model_name_or_path
    retr_tokenizer = AutoTokenizer.from_pretrained(retr_model_name)
    retr_model_base = AutoModel.from_pretrained(retr_model_name).to("cuda")
    infer_tokenizer = AutoTokenizer.from_pretrained(infer_model_name, trust_remote_code=True)
    if infer_tokenizer.pad_token is None:
        logger.info("No padding token - using EOS instead")
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
    infer_tokenizer.padding_side = 'left'
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

    eval_steps_fraction = getattr(args, 'eval_steps', None)

    if log_wandb:
        wandb_key = os.getenv('WANDB_KEY')
        name = args.wandb_proj_name
        wandb.login(key=wandb_key)
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
            "dataset_path": args.dataset_path,
        }
        if eval_steps_fraction is not None:
            wandb_config["eval_steps_fraction"] = eval_steps_fraction
        wandb.init(
            project='fine-tuning-retriever',
            name=name if name else f"training run - {args.dataset_path.split('/')[-1]}",
            config=wandb_config
        )
        wandb.watch(retr_model_base, log_freq=10)

    infer_model.eval()
    retr_model_base.eval()

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

    optimizer = AdamW(retr_model_base.parameters(), lr=args.learning_rate)
    num_training_steps = num_epochs * len(train_data_loader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    cross_entropy = CrossEntropyLoss(reduction='none')
    kl_div = KLDivLoss(reduction='none')

    inner_data_collator = DataCollatorForCompletionOnlyLM(
        response_template=ANSWER, 
        tokenizer=infer_tokenizer, 
        mlm=False)

    def inner_tokenize_function(samples):
        return infer_tokenizer(samples["text"], truncation=True, padding=True, return_tensors="pt")
    
    def parse_batch(dataset_split, batch, index):
        curr_bs = batch["input_ids"].size(0)
        batch_data = dataset_split[index*batch_size:(index*batch_size)+curr_bs]
        return curr_bs, batch_data

    def get_top_k_docs_per_query(embedded_documents, batch_data, k, current_retr_model):
        embedded_queries = current_retr_model.encode(batch_data[query_column], 
                                                     convert_to_tensor=True, 
                                                     show_progress_bar=False)
        embedded_documents_exp = embedded_documents.unsqueeze(0)
        embedded_queries_exp = embedded_queries.unsqueeze(1)
        cos_sim = F.cosine_similarity(embedded_documents_exp, embedded_queries_exp, dim=-1)
        top_k_docs = torch.topk(cos_sim, k, dim=-1)
        return top_k_docs.indices, top_k_docs.values, cos_sim

    compute_Pr = lambda similarities, gamma: F.softmax(similarities / gamma, dim=1)

    def get_prompts(prompt_template, documents, batch_data, documents_per_query):
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
        Q_log = torch.log(Q)
        divergence = kl_div(Q_log, Pr).sum(-1)
        loss = divergence.mean()
        return loss

    def evaluate_with_retrieval_evaluator(retr_model_eval: SentenceTransformer,
                                          eval_dataset_split: Dataset,
                                          eval_api_corpus: list,
                                          corpus_embeddings: torch.Tensor,
                                          device: str = "cuda",
                                          k_values_accuracy: list = [1, 3, 5],
                                          k_values_ndcg: list = [1, 3, 5],
                                          eval_batch_size: int = 16):
        logger.info("Preparing data for DeviceAwareInformationRetrievalEvaluator")

        queries = {}
        relevant_docs = {}
        corpus = {str(idx): doc for idx, doc in enumerate(eval_api_corpus)}
        api_to_doc_id = {api_desc: str(idx) for idx, api_desc in enumerate(eval_api_corpus)}

        for i, example in enumerate(eval_dataset_split):
            query_text = example[query_column]
            positive_apis = example["api_description"]
            
            if not isinstance(positive_apis, list):
                positive_apis = [positive_apis]

            query_id = f"query_{i}"
            queries[query_id] = query_text
            
            current_relevant_ids = set()
            for pos_api in positive_apis:
                 if pos_api in api_to_doc_id:
                     current_relevant_ids.add(api_to_doc_id[pos_api])
                 else:
                      logger.warning(f"Positive API '{pos_api[:50]}...' not found in corpus map for query '{query_text[:50]}...'")
            
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

        evaluator = DeviceAwareInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            ndcg_at_k=k_values_ndcg,
            accuracy_at_k=k_values_accuracy,
            device=device,
            batch_size=eval_batch_size,
            show_progress_bar=True,
        )

        logger.info("Running evaluation with DeviceAwareInformationRetrievalEvaluator")
        
        scores = evaluator.compute_metrices(
            model=retr_model_eval,
            corpus_embeddings=corpus_embeddings
        )

        if log_wandb:
            flat_scores = {}
            for score_type, metrics in scores.items():
                for metric_name, values in metrics.items():
                    if isinstance(values, dict):
                        for k_val, score in values.items():
                            flat_scores[f"eval_{score_type}_{metric_name}_{k_val}"] = score
                    else:
                         flat_scores[f"eval_{score_type}_{metric_name}"] = values
            wandb.log(flat_scores)
            
        logger.info("Evaluation Results:")
        for score_type, metrics in scores.items():
            logger.info(f"  Score Type: {score_type}")
            for metric_name, values in metrics.items():
                 if isinstance(values, dict):
                     for k_val, score in values.items():
                         logger.info(f"    {metric_name}@{k_val}: {score:.4f}")
                 else:
                     logger.info(f"    {metric_name}: {values:.4f}")

        return scores

    def get_sentence_transformer(base_model, device="cuda"):
        pooling_model = models.Pooling(base_model.config.hidden_size, pooling_mode_cls_token=True)
        return SentenceTransformer(modules=[base_model, pooling_model], device=device)

    steps_per_epoch = len(train_data_loader)
    evaluation_step_interval = None
    run_end_of_epoch_eval = True

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
    retr_model_eval_instance = get_sentence_transformer(retr_model_base, device="cuda")
    with torch.no_grad():
        embedded_eval_documents = embed_corpus(retr_model_eval_instance,
                                               retr_tokenizer,
                                               eval_documents,
                                               device="cuda",
                                               batch_size=args.batch_size,
                                               max_length=args.retr_max_seq_length)

    evaluate_with_retrieval_evaluator(
        retr_model_eval=retr_model_eval_instance,
        eval_dataset_split=dataset["test"],
        eval_api_corpus=eval_documents,
        corpus_embeddings=embedded_eval_documents,
        device="cuda",
        k_values_accuracy=[1, 3, 5],
        k_values_ndcg=[1, 3, 5],
        eval_batch_size=args.batch_size
    )
    del embedded_eval_documents, retr_model_eval_instance
    torch.cuda.empty_cache()

    retr_model_base.train()
    global_step = 0
    for epoch in range(num_epochs):
        for index, batch in enumerate(train_data_loader):
            current_step_in_epoch = index + 1
            global_step += 1
            if verbose:
                logger.info(f"Epoch: {epoch}, Batch: {index}")
            curr_bs, batch_data = parse_batch(dataset["train"], batch, index)

            retr_model_train_instance = get_sentence_transformer(retr_model_base, device="cuda")
            retr_model_train_instance.eval()
            
            with torch.no_grad():
                embedded_documents = embed_corpus(retr_model_train_instance,
                                                  retr_tokenizer,
                                                  train_documents,
                                                  device="cuda",
                                                  batch_size=args.batch_size,
                                                  max_length=args.retr_max_seq_length)

            documents_per_query, similarities_per_query, _ = get_top_k_docs_per_query(
                embedded_documents, batch_data, k, retr_model_train_instance
            )

            retr_model_base.train()
            
            Pr = compute_Pr(similarities_per_query, gamma)
            del similarities_per_query, _, embedded_documents, retr_model_train_instance
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
            del perplexities, Q, Pr, inner_data_loader, inner_batch, documents_per_query
            torch.cuda.empty_cache()
            if log_wandb:
                wandb.log({"Training Loss": loss.item(), "Epoch": epoch, "Step": global_step})

            if evaluation_step_interval is not None and global_step % evaluation_step_interval == 0:
                logger.info(f"--- Evaluating at Epoch {epoch+1}, Step {current_step_in_epoch}/{steps_per_epoch} (Global Step {global_step}) ---")
                retr_model_base.eval()
                retr_model_eval_instance = get_sentence_transformer(retr_model_base, device="cuda")
                with torch.no_grad():
                    embedded_eval_documents = embed_corpus(retr_model_eval_instance,
                                                           retr_tokenizer,
                                                           eval_documents,
                                                           device="cuda",
                                                           batch_size=args.batch_size,
                                                           max_length=args.retr_max_seq_length)

                evaluate_with_retrieval_evaluator(
                    retr_model_eval=retr_model_eval_instance,
                    eval_dataset_split=dataset["test"],
                    eval_api_corpus=eval_documents,
                    corpus_embeddings=embedded_eval_documents,
                    device="cuda",
                    k_values_accuracy=[1, 3, 5],
                    k_values_ndcg=[1, 3, 5],
                    eval_batch_size=args.batch_size
                )
                retr_model_base.train()
                del embedded_eval_documents, retr_model_eval_instance
                torch.cuda.empty_cache()

        if run_end_of_epoch_eval:
            logger.info(f"--- Evaluating after Epoch {epoch+1} ---")
            retr_model_base.eval()
            retr_model_eval_instance = get_sentence_transformer(retr_model_base, device="cuda")
            with torch.no_grad():
                embedded_eval_documents = embed_corpus(retr_model_eval_instance,
                                                       retr_tokenizer,
                                                       eval_documents,
                                                       device="cuda",
                                                       batch_size=args.batch_size,
                                                       max_length=args.retr_max_seq_length)

            evaluate_with_retrieval_evaluator(
                retr_model_eval=retr_model_eval_instance,
                eval_dataset_split=dataset["test"],
                eval_api_corpus=eval_documents,
                corpus_embeddings=embedded_eval_documents,
                device="cuda",
                k_values_accuracy=[1, 3, 5],
                k_values_ndcg=[1, 3, 5],
                eval_batch_size=args.batch_size
            )
            retr_model_base.train()
            del embedded_eval_documents, retr_model_eval_instance
            torch.cuda.empty_cache()

    if verbose:
        logger.info("TRAINING FINISHED.")
    if log_wandb:
        wandb.finish()
    if args.trained_model_save_path:
        retr_model_base.save_pretrained(args.trained_model_save_path)
        retr_tokenizer.save_pretrained(args.trained_model_save_path)
        logger.info(f"Saved trained base model to {args.trained_model_save_path}")

if __name__ == "__main__":
    main()

