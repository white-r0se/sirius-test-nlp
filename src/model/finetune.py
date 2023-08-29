import wandb
import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(cur_path)
sys.path.append(os.path.join(parent_path, "data"))
from dataset import ApplyWordDropout, ConversationDataset, ConversationDataModule


class Config:
    def __init__(self):
        self.seed = 42
        self.word_dropout = 0
        self.val_size = 0.2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optim = "adamw_torch"
        self.output_dir = "./output"
        self.report_to = "wandb"
        self.overwrite_output_dir = True
        self.per_device_train_batch_size = 1
        self.per_device_eval_batch_size = 1
        self.gradient_accumulation_steps = 128
        self.num_train_epochs = 5
        self.max_steps = -1
        self.save_steps = 100
        self.evaluation_strategy = "steps"
        self.eval_steps = 40
        self.logging_steps = 2
        self.fp16 = True
        self.warmup_ratio = 0.1
        self.learning_rate = 1e-4
        self.prediction_loss_only = True
        self.path_to_data = str(
            os.path.join(
                os.path.dirname(parent_path), "data/processed/data_processed.csv"
            )
        )
        self.path_to_save = str(
            os.path.join(
                os.path.dirname(parent_path),
                f"models/rudialogpt-medium-lora-{self.num_train_epochs}ep",
            )
        )


class FineTune:
    def __init__(self, config, tokenizer_name, model_name):
        self.config = config
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.lora_config = None
        self.data_collator = None
        self.training_args = None
        self.data_module = None
        self.trainer = None

    def set_seed(self):
        np.random.seed(self.config.seed)
        os.environ["PYTHONHASHSEED"] = str(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_model(self):
        self.set_seed()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            self.config.device
        )
        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["wte", "lm_head"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        self.training_args = TrainingArguments(
            optim=self.config.optim,
            output_dir=self.config.output_dir,
            report_to=self.config.report_to,
            overwrite_output_dir=self.config.overwrite_output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16,
            prediction_loss_only=self.config.prediction_loss_only,
        )
        self.data_module = ConversationDataModule(
            pd.read_csv(self.config.path_to_data).fillna(""), self.config
        )
        self.data_module.setup()
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.data_module.train_dataset,
            eval_dataset=self.data_module.val_dataset,
        )

    def fine_tune(self):
        if self.trainer is None:
            self.prepare_model()
        self.trainer.train()
        self.trainer.save_model(self.config.path_to_save)


def main():
    wandb.init(project="sirius-nlp-chatbot")
    config = Config()
    fine_tune = FineTune(
        config, "tinkoff-ai/ruDialoGPT-medium", "tinkoff-ai/ruDialoGPT-medium"
    )
    fine_tune.prepare_model()
    fine_tune.fine_tune()
    wandb.finish()


if __name__ == "__main__":
    main()
