import os
import torch
from pydantic import BaseModel

cur_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(cur_path)


class Config(BaseModel):
    seed: int = 42
    word_dropout: float = 0
    val_size: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    optim: str = "adamw_torch"
    output_dir: str = "./output"
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 128
    num_train_epochs: int = 5
    max_steps: int = -1
    save_steps: int = 100
    evaluation_strategy: str = "steps"
    eval_steps: int = 40
    logging_steps: int = 2
    fp16: bool = True
    warmup_ratio: float = 0.1
    learning_rate: float = 1e-4
    prediction_loss_only: bool = True
    path_to_data: str = str(
        os.path.join(os.path.dirname(parent_path), "data/processed/data_processed.csv")
    )
    path_to_save: str = str(
        os.path.join(
            os.path.dirname(parent_path),
            f"models/rudialogpt-medium-lora-{num_train_epochs}ep",
        )
    )
