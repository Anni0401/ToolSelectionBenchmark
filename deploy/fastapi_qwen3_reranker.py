"""OpenAI-compatible FastAPI score server for Qwen3-Reranker.

Implements the endpoint expected by langgraph_app.py:
    POST /v1/score

The request format is compatible with vLLM's score API:
    {"model": str, "text_1": str, "text_2": str | list[str],
     "encoding_format": "float"}
"""

from __future__ import annotations

import os
import traceback
from threading import Lock
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-Reranker-8B")
MODEL_CACHE_DIR = os.getenv("CHECKPOINT_DIR", os.path.expanduser("~/.cache/huggingface/hub"))
MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))
BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "1"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_dtype_name = os.getenv("DTYPE", "bfloat16").lower()
if DEVICE == "cpu":
    DTYPE = torch.float32
elif _dtype_name in {"bfloat16", "bf16"}:
    DTYPE = torch.bfloat16
elif _dtype_name in {"float16", "fp16", "half"}:
    DTYPE = torch.float16
else:
    raise ValueError(f"Unsupported DTYPE={_dtype_name!r}")

TASK_INSTRUCTION = (
    "Given a user query about tool usage, retrieve the most relevant tool "
    "function that can fulfill the described task."
)
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
)


class ScoreRequest(BaseModel):
    model: str | None = None
    text_1: str
    text_2: str | list[str]
    encoding_format: str | None = "float"


class ScoreItem(BaseModel):
    index: int
    score: float


class ScoreResponse(BaseModel):
    object: str = "list"
    data: list[ScoreItem]
    model: str


def _prompt(query: str, document: str) -> str:
    """Build the same Qwen3 reranker prompt used by the client fallback."""
    return (
        f"<Instruct>: {TASK_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}\n"
        "Does the Document meet the requirements? Respond with 'Yes' or 'No'."
    )


class Reranker:
    def __init__(self) -> None:
        print(f"[RERANKER] Loading {MODEL_NAME} on {DEVICE} ({DTYPE})")
        self._inference_lock = Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True,
            local_files_only=True,
        )
        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True,
            local_files_only=True,
        )
        # Qwen3-Reranker checkpoints may omit a pad token. Batched
        # sequence-classification requires one for left/right padding.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # The public Qwen3-Reranker checkpoint is a causal LM. Its reranking
        # score is the probability of the first Yes/No token. Only use the
        # classification loader for checkpoints explicitly published with a
        # sequence-classification architecture; otherwise a fresh classifier
        # head can produce nearly identical scores for every document.
        architectures = " ".join(config.architectures or [])
        if "ForSequenceClassification" in architectures:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                torch_dtype=DTYPE,
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
            self.model_kind = "classification"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                torch_dtype=DTYPE,
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
            self.model_kind = "causal"
            # Use the official Qwen3-Reranker prefix/suffix. In particular,
            # the <think>...</think> suffix is required: logits at the end of
            # a plain chat-template prompt are not the trained yes/no position.
            self.tokenizer.padding_side = "left"
            self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
            self.prefix_tokens = self.tokenizer.encode(
                "<|im_start|>system\n"
                f"{SYSTEM_PROMPT}<|im_end|>\n"
                "<|im_start|>user\n",
                add_special_tokens=False,
            )
            self.suffix_tokens = self.tokenizer.encode(
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                add_special_tokens=False,
            )
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        self.model.to(DEVICE)
        self.model.eval()
        print("[RERANKER] Model ready")

    def score(self, query: str, documents: list[str]) -> list[float]:
        with self._inference_lock:
            return self._score_locked(query, documents)

    @torch.inference_mode()
    def _score_locked(self, query: str, documents: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(documents), BATCH_SIZE):
            batch_docs = documents[start : start + BATCH_SIZE]
            if self.model_kind == "causal":
                bodies = [_prompt(query, doc) for doc in batch_docs]
                body_inputs = self.tokenizer(
                    bodies,
                    padding=False,
                    truncation=True,
                    max_length=max(1, MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens)),
                    return_attention_mask=False,
                )
                batch_inputs = []
                for input_ids in body_inputs["input_ids"]:
                    batch_inputs.append(
                        {"input_ids": self.prefix_tokens + input_ids + self.suffix_tokens}
                    )
                encoded = self.tokenizer.pad(batch_inputs, padding=True, return_tensors="pt")
                encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
                output = self.model(**encoded, return_dict=True, use_cache=False)
                logits = output.logits[:, -1, :]
                true_logits = logits[:, self.true_token_id]
                false_logits = logits[:, self.false_token_id]
                batch_scores = torch.softmax(
                    torch.stack([false_logits, true_logits], dim=1), dim=1
                )[:, 1]
                scores.extend(float(value) for value in batch_scores.detach().cpu())
                del encoded, output, logits, true_logits, false_logits, batch_scores
                continue

            texts = []
            for doc in batch_docs:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _prompt(query, doc)},
                ]
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except (AttributeError, ValueError):
                    # Some older tokenizer revisions do not ship a chat
                    # template. The model can still score the plain reranker
                    # prompt, and this keeps the endpoint usable.
                    text = f"{SYSTEM_PROMPT}\n\n{_prompt(query, doc)}"
                texts.append(text)
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            output = self.model(**encoded, return_dict=True, use_cache=False)
            logits = output.logits
            if logits.ndim == 3:
                # Be tolerant of causal-model style sequence logits: use the
                # final non-padding position for the yes/no classification.
                last_positions = encoded["attention_mask"].sum(dim=1) - 1
                logits = logits[
                    torch.arange(logits.shape[0], device=logits.device),
                    last_positions,
                ]
            if logits.ndim != 2 or logits.shape[-1] < 1:
                raise RuntimeError(f"Unexpected reranker logits shape: {tuple(logits.shape)}")

            # Qwen3-Reranker is a two-label classifier. Prefer the explicit
            # yes/no labels; sigmoid(logit_yes - logit_no) is numerically stable.
            if logits.shape[-1] >= 2:
                batch_scores = torch.sigmoid(logits[:, 1] - logits[:, 0])
            else:
                batch_scores = torch.sigmoid(logits[:, 0])
            scores.extend(float(value) for value in batch_scores.detach().cpu())
            del encoded, output, logits, batch_scores
        return scores


reranker: Reranker | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global reranker
    reranker = Reranker()
    yield
    reranker = None


app = FastAPI(title="Qwen3 Reranker Score API", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    if reranker is None:
        raise HTTPException(status_code=503, detail="Reranker is still loading")
    documents = [request.text_2] if isinstance(request.text_2, str) else request.text_2
    if not documents:
        return ScoreResponse(model=request.model or MODEL_NAME, data=[])
    try:
        scores = reranker.score(request.text_1, documents)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ScoreResponse(
        model=request.model or MODEL_NAME,
        data=[ScoreItem(index=index, score=value) for index, value in enumerate(scores)],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_qwen3_reranker:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8003")),
        workers=1,
    )