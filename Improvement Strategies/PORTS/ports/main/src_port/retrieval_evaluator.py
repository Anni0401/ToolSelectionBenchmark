from __future__ import annotations

import heapq
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm import trange

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class DeviceAwareInformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting with device awareness.
    It ensures all tensors are on the same device during evaluation.

    Given a set of queries and a large corpus set, it will retrieve for each query the top-k most similar document. 
    It measures Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(
        self,
        queries: dict[str, str],  # qid => query
        corpus: dict[str, str],  # cid => doc
        relevant_docs: dict[str, set[str]],  # qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
        corpus_prompt: str | None = None,
        corpus_prompt_name: str | None = None,
        device: Optional[str] = None,  # Added device parameter
    ) -> None:
        """
        Initializes the InformationRetrievalEvaluator with device awareness.

        Args:
            queries (Dict[str, str]): A dictionary mapping query IDs to queries.
            corpus (Dict[str, str]): A dictionary mapping document IDs to documents.
            relevant_docs (Dict[str, Set[str]]): A dictionary mapping query IDs to a set of relevant document IDs.
            corpus_chunk_size (int): The size of each chunk of the corpus. Defaults to 50000.
            mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation.
            ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation.
            accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation.
            precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation.
            map_at_k (List[int]): A list of integers representing the values of k for MAP calculation.
            show_progress_bar (bool): Whether to show a progress bar during evaluation.
            batch_size (int): The batch size for evaluation.
            name (str): A name for the evaluation.
            write_csv (bool): Whether to write the evaluation results to a CSV file.
            truncate_dim (int, optional): The dimension to truncate the embeddings to.
            score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions.
            main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation.
            query_prompt (str, optional): The prompt to be used when encoding the corpus.
            query_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus.
            corpus_prompt (str, optional): The prompt to be used when encoding the corpus.
            corpus_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus.
            device (str, optional): The device to run evaluations on ('cpu' or 'cuda').
        """
        super().__init__()
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]
        
        # Validate that we have queries to process
        if len(self.queries) == 0:
            raise ValueError("No valid queries found. Please check the 'queries' and 'relevant_docs' parameters.")

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        
        # Validate that we have corpus entries
        if len(self.corpus) == 0:
            raise ValueError("Corpus is empty. Please provide a valid corpus dictionary.")

        self.query_prompt = query_prompt
        self.query_prompt_name = query_prompt_name
        self.corpus_prompt = corpus_prompt
        self.corpus_prompt_name = corpus_prompt_name

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None
        self.truncate_dim = truncate_dim
        
        # Store the device
        self.device = device

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        self._append_csv_headers(self.score_function_names)

    def _append_csv_headers(self, score_function_names):
        # Same implementation as original
        for score_name in score_function_names:
            for k in self.accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")

            for k in self.precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k}")
                self.csv_headers.append(f"{score_name}-Recall@{k}")

            for k in self.mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")

            for k in self.ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")

            for k in self.map_at_k:
                self.csv_headers.append(f"{score_name}-MAP@{k}")

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        # If device isn't set, use the model's device
        if self.device is None:
            self.device = next(model.parameters()).device
            logger.info(f"Using model's device for evaluation: {self.device}")

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Information Retrieval Evaluation of the model on the {self.name} dataset{out_txt}:")

        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")
            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [(name, scores[name]["ndcg@k"][max(self.ndcg_at_k)]) for name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"

        metrics = {
            f"{score_function}_{metric_name.replace('@k', '@' + str(k))}": value
            for score_function, values_dict in scores.items()
            for metric_name, values in values_dict.items()
            for k, value in values.items()
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def compute_metrices(
        self, model: SentenceTransformer, corpus_model=None, corpus_embeddings: Tensor | None = None
    ) -> dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        # Validate queries and corpus
        if not self.queries or len(self.queries) == 0:
            logger.error("No queries available for evaluation")
            return {"error": "No queries available"}
        
        if not self.corpus or len(self.corpus) == 0:
            logger.error("No corpus available for evaluation")
            return {"error": "No corpus available"}
            
        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Get the device for model
        device = self.device if self.device is not None else next(model.parameters()).device
        logger.info(f"Computing metrics using device: {device}")

        # Compute embedding for the queries
        logger.info(f"Encoding {len(self.queries)} queries...")
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            query_embeddings = model.encode(
                self.queries,
                prompt_name=self.query_prompt_name,
                prompt=self.query_prompt,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
                device=device
            )

        # Validate query embeddings shape
        if query_embeddings.shape[0] == 0 or query_embeddings.shape[1] == 0:
            logger.error(f"Query embeddings have invalid shape: {query_embeddings.shape}")
            return {"error": f"Query embeddings have invalid shape: {query_embeddings.shape}"}
        else:
            logger.info(f"Query embeddings shape: {query_embeddings.shape}")

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Iterate over chunks of the corpus
        logger.info(f"Processing corpus in chunks of {self.corpus_chunk_size} documents...")
        for corpus_start_idx in trange(
            0, len(self.corpus), self.corpus_chunk_size, desc="Corpus Chunks", disable=not self.show_progress_bar
        ):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            # Encode chunk of corpus
            if corpus_embeddings is None:
                logger.info(f"Encoding corpus chunk {corpus_start_idx}:{corpus_end_idx}...")
                with (
                    nullcontext()
                    if self.truncate_dim is None
                    else corpus_model.truncate_sentence_embeddings(self.truncate_dim)
                ):
                    sub_corpus_embeddings = corpus_model.encode(
                        self.corpus[corpus_start_idx:corpus_end_idx],
                        prompt_name=self.corpus_prompt_name,
                        prompt=self.corpus_prompt,
                        batch_size=self.batch_size,
                        show_progress_bar=self.show_progress_bar,
                        convert_to_tensor=True,
                        device=device  # Explicitly set device
                    )
            else:
                # If corpus embeddings are provided, make sure they're on the right device
                sub_corpus_slice = corpus_embeddings[corpus_start_idx:corpus_end_idx]
                if sub_corpus_slice.shape[0] == 0:
                    logger.warning(f"Empty corpus slice at {corpus_start_idx}:{corpus_end_idx}, skipping")
                    continue
                sub_corpus_embeddings = sub_corpus_slice.to(device)
            
            # Validate corpus embeddings
            if sub_corpus_embeddings.shape[0] == 0:
                logger.warning(f"Corpus embeddings for chunk {corpus_start_idx}:{corpus_end_idx} are empty, skipping")
                continue
            
            logger.info(f"Corpus chunk embeddings shape: {sub_corpus_embeddings.shape}")

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                # Ensure both tensors are on the same device
                if query_embeddings.device != sub_corpus_embeddings.device:
                    logger.warning(f"Device mismatch detected: queries on {query_embeddings.device}, corpus on {sub_corpus_embeddings.device}. Moving corpus to {query_embeddings.device}")
                    sub_corpus_embeddings = sub_corpus_embeddings.to(query_embeddings.device)

                try:
                    # Now compute scores - Fix for dimension mismatch
                    logger.info(f"Computing similarity scores between queries {query_embeddings.shape} and corpus {sub_corpus_embeddings.shape}")
                    
                    # Calculate cosine similarity properly with batch processing
                    if name == "cosine":
                        # Create a matrix of scores (queries x corpus)
                        pair_scores = torch.zeros(len(query_embeddings), len(sub_corpus_embeddings), device=device)
                        
                        # Process in smaller batches to avoid memory issues
                        batch_size = self.batch_size
                        for i in range(0, len(query_embeddings), batch_size):
                            end_idx = min(i + batch_size, len(query_embeddings))
                            q_batch = query_embeddings[i:end_idx]
                            
                            # For each query in the batch, calculate similarity with all corpus items
                            for j in range(len(sub_corpus_embeddings)):
                                # Get corpus embedding and expand to match batch size
                                c_emb = sub_corpus_embeddings[j:j+1]  # Keep dimension
                                
                                # Calculate similarity
                                similarity = F.cosine_similarity(
                                    q_batch, 
                                    c_emb.expand(end_idx - i, -1),  # Expand to match batch size
                                    dim=1
                                )
                                pair_scores[i:end_idx, j] = similarity
                    else:
                        # For any other function that might handle the dimensions properly
                        pair_scores = score_function(query_embeddings, sub_corpus_embeddings)
                    
                    # Get top-k values
                    pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                        pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
                    )
                    pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                    pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                    for query_itr in range(len(query_embeddings)):
                        for sub_corpus_id, score in zip(
                            pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]
                        ):
                            corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                            if len(queries_result_list[name][query_itr]) < max_k:
                                heapq.heappush(queries_result_list[name][query_itr], (score, corpus_id))
                            else:
                                heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))
                except RuntimeError as e:
                    logger.error(f"Error computing similarity scores: {e}")
                    logger.error(f"Query embeddings shape: {query_embeddings.shape}, Corpus embeddings shape: {sub_corpus_embeddings.shape}")
                    raise RuntimeError(f"Error computing similarity scores: {e}. Check embedding dimensions.")

        # Validate result list 
        if all(len(result_list) == 0 for name in queries_result_list for result_list in queries_result_list[name]):
            logger.warning("No results found for any query. Check your data and embeddings.")
            # Return default metrics with zero values
            default_scores = {}
            for name in self.score_functions:
                default_scores[name] = {
                    "accuracy@k": {k: 0.0 for k in self.accuracy_at_k},
                    "precision@k": {k: 0.0 for k in self.precision_recall_at_k},
                    "recall@k": {k: 0.0 for k in self.precision_recall_at_k},
                    "ndcg@k": {k: 0.0 for k in self.ndcg_at_k},
                    "mrr@k": {k: 0.0 for k in self.mrr_at_k},
                    "map@k": {k: 0.0 for k in self.map_at_k},
                }
            return default_scores

        # Process the results
        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {"corpus_id": corpus_id, "score": score}

        logger.info(f"Queries: {len(self.queries)}")
        logger.info(f"Corpus: {len(self.corpus)}\n")

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores

    # The rest of the class implementation remains unchanged
    def compute_metrics(self, queries_result_list: list[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    def output_scores(self, scores):
        for k in scores["accuracy@k"]:
            logger.info("Accuracy@{}: {:.2f}%".format(k, scores["accuracy@k"][k] * 100))

        for k in scores["precision@k"]:
            logger.info("Precision@{}: {:.2f}%".format(k, scores["precision@k"][k] * 100))

        for k in scores["recall@k"]:
            logger.info("Recall@{}: {:.2f}%".format(k, scores["recall@k"][k] * 100))

        for k in scores["mrr@k"]:
            logger.info("MRR@{}: {:.4f}".format(k, scores["mrr@k"][k]))

        for k in scores["ndcg@k"]:
            logger.info("NDCG@{}: {:.4f}".format(k, scores["ndcg@k"][k]))

        for k in scores["map@k"]:
            logger.info("MAP@{}: {:.4f}".format(k, scores["map@k"][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg