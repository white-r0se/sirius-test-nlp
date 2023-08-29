import unittest
from unittest.mock import Mock, patch
import pandas as pd
import re
import os
import transformers
from ..model.finetune import (
    Config,
    FineTune,
    ConversationDataModule,
    LoraConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class TestFineTune(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.fp16 = False
        self.tokenizer_name = "tinkoff-ai/ruDialoGPT-medium"
        self.model_name = "tinkoff-ai/ruDialoGPT-medium"
        self.fine_tune_model = FineTune(
            self.config, self.tokenizer_name, self.model_name
        )

    def test_prepare_model(self):
        self.fine_tune_model.prepare_model()
        assert self.fine_tune_model.tokenizer is not None
        assert isinstance(
            self.fine_tune_model.tokenizer,
            transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast,
        )
        assert self.fine_tune_model.model is not None
        assert isinstance(
            self.fine_tune_model.model,
            transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel,
        )
        assert self.fine_tune_model.lora_config is not None
        assert isinstance(self.fine_tune_model.lora_config, LoraConfig)
        assert self.fine_tune_model.data_collator is not None
        assert isinstance(
            self.fine_tune_model.data_collator, DataCollatorForLanguageModeling
        )
        assert self.fine_tune_model.training_args is not None
        assert isinstance(self.fine_tune_model.training_args, TrainingArguments)
        assert self.fine_tune_model.data_module is not None
        assert isinstance(self.fine_tune_model.data_module, ConversationDataModule)
        assert self.fine_tune_model.trainer is not None
        assert isinstance(self.fine_tune_model.trainer, Trainer)

    def test_fine_tune(self):
        self.fine_tune_model = FineTune(
            self.config, self.tokenizer_name, self.model_name
        )
        self.fine_tune_model.config.path_to_data = re.sub(
            r"processed.csv", "test.csv", self.fine_tune_model.config.path_to_data
        )
        self.fine_tune_model.config.fp16 = False
        self.fine_tune_model.config.num_train_epochs = 0.001
        self.fine_tune_model.config.max_steps = 1
        self.fine_tune_model.config.gradient_accumulation_steps = 1
        self.fine_tune_model.config.path_to_save = (
            self.fine_tune_model.config.path_to_save + "-test"
        )
        self.fine_tune_model.prepare_model()
        self.fine_tune_model.fine_tune()
        assert os.path.exists(self.fine_tune_model.config.path_to_save)


if __name__ == "__main__":
    unittest.main()
