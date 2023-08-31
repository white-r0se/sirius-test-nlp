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
from src.data.dataset import ConversationDataModule
from src.config.config_for_trainer import Config


class TrainerFineTune:
    def __init__(self, config: Config, tokenizer_name: str, model_name: str):
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
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, self.lora_config)
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
    with open("src/config/config.json", "r") as f:
        config = Config().model_validate_json(f.read())
    fine_tune = TrainerFineTune(
        config, "tinkoff-ai/ruDialoGPT-medium", "tinkoff-ai/ruDialoGPT-medium"
    )
    fine_tune.prepare_model()
    with wandb.init(project="lora-finetune-rudialogpt", config=config):
        fine_tune.fine_tune()


if __name__ == "__main__":
    main()
