import wandb
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(cur_path)
sys.path.append(os.path.join(parent_path, "data"))
from dataset import ApplyWordDropout, ConversationDataset, ConversationDataModule

class Config():
    def __init__(self):
        self.seed = 42
        self.word_dropout = 0
        self.val_size = 0.2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optim = "adamw_torch"
        self.output_dir = "./output"
        self.report_to = "wandb"
        self.overwrite_output_dir = True,
        self.per_device_train_batch_size = 1,
        self.per_device_eval_batch_size = 1,
        self.gradient_accumulation_steps = 128,
        self.num_train_epochs = 5,
        self.save_steps = 100,
        self.evaluation_strategy = "steps",
        self.eval_steps = 60,
        self.logging_steps = 5,
        self.fpt16 = "int8",
        self.warmup_steps = 100,
        self.prediction_loss_only = True,
        self.path_to_data = os.path.join(os.path.dirname(parent_path), "data/processed/data.csv")
        self.path_to_save = os.path.join(os.path.dirname(parent_path), "models/rudialogpt-medium-lora-{self.max_epochs}ep}")

class FineTune:
    def __init__(self, config, tokenizer_name, model_name, data_path):
        self.config = config
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.data_path = data_path

    def set_seed(self):
        np.random.seed(self.config.seed)
        os.environ["PYTHONHASHSEED"] = str(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_model(self):
        self.set_seed(self.config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, padding_side='left'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        ).to(self.config.device)
        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["wte", "lm_head"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

    def fine_tune(self):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        training_args = TrainingArguments(
            optim="adamw_torch",
            output_dir="./output",
            report_to="wandb",
            evaluation_strategy='steps',
            eval_steps=3000,
            overwrite_output_dir=True,
            gradient_accumulation_steps=2,
            num_train_epochs=5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            save_steps=500,
            logging_steps=100,
            fp16=True,
            prediction_loss_only=True
        )

        data = pd.read_csv(self.data_path)
        conversation_data_module = ConversationDataModule(
            data, self.config
        )
        conversation_data_module.setup()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=conversation_data_module.train_dataset,
            eval_dataset=conversation_data_module.val_dataset,
        )
        trainer.train()
        trainer.save_model(self.config.path_to_save)

def main():
    wandb.init(project="sirius-nlp-chatbot")
    config = Config()
    fine_tune = FineTune(
        config, "tinkoff-ai/ruDialoGPT-small", "tinkoff-ai/ruDialoGPT-medium", config.path_to_data
    )
    fine_tune.prepare_model()
    fine_tune.fine_tune()
    wandb.finish()

if __name__ == '__main__':
    main()
