from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PyTorchTrainingParams:

    dataset_path: str = field(
        metadata={"help": "The path of the training dataset."}
    )

    retr_model_name_or_path: str = field(
        default="BAAI/bge-base-en-v1.5",
        metadata={"help": "Path to pretrained retrieval model or model identifier from huggingface.co/models"}
    )

    infer_model_name_or_path: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Path to pretrained inference model or model identifier from huggingface.co/models"}
    )

    infer_model_type: str = field(
        default="llama3",
        metadata={"help": "The type of the model to train."}
    )

    query_column: str = field(
        default="query",
        metadata={"help": "The name of the query column in the dataset."},
    )

    response_column: str = field(
        default="response",
        metadata={"help": "The name of the response column in the dataset."},
    )

    batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})

    num_retrieved_docs_per_query: int = field(default=3, metadata={"help": "Number of retrieved documents per query (size of D' in RePlug paper)."})

    gamma_value: float = field(default=1.0, metadata={"help": "Gamma value for the loss function."})

    beta_value: float = field(default=1.0, metadata={"help": "Beta value for the loss function."})

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})

    lr_scheduler: str = field(
        default='cosine',
        metadata={"help" : "Type of learning rate scheduler."}
    )

    trained_model_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the trained model."}
    )

    quantize: bool = field(
        default=False,
        metadata={"help": "Whether to load the model quantized."}
    )

    quantization_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4-bit quantization or 8-bit quantization."}
    )

    log_to_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to log training progress to Weights & Biases."}
    )

    wandb_proj_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name."}
    )

    verbose: bool = field(
        default=False,
        metadata={"help": "Whether to print training logs."}
    )
    
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."}
    )
