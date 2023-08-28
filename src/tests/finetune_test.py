import unittest
from unittest.mock import Mock, patch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, PeftModel
import os
import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(cur_path)
sys.path.append(os.path.join(parent_path, "model"))
from finetune import Config, FineTune

class TestFineTune(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.tokenizer_name = "tinkoff-ai/ruDialoGPT-medium"
        self.model_name = "tinkoff-ai/ruDialoGPT-medium"
        self.fine_tune_model = FineTune(self.config, self.tokenizer_name, self.model_name)

    @patch('finetune.Trainer')
    @patch('finetune.ConversationDataModule')
    @patch('finetune.TrainingArguments')
    @patch('finetune.DataCollatorForLanguageModeling')
    @patch('finetune.LoraConfig')
    @patch('finetune.AutoTokenizer.from_pretrained')
    @patch('finetune.AutoModelForCausalLM.from_pretrained')
    def test_prepare_model(self, mock_model, mock_tokenizer, mock_config, mock_data_collator, mock_training_args, mock_data_module, mock_trainer):
        self.fine_tune_model.prepare_model()

        mock_tokenizer.assert_called_once_with(self.tokenizer_name, padding_side='left')
        mock_model.assert_called_once_with(self.model_name)
        mock_model.return_value.to.assert_called_once_with(self.config.device)
        mock_config.assert_called_once()
        mock_data_collator.assert_called_once_with(tokenizer=mock_tokenizer.return_value, mlm=False)
        mock_training_args.assert_called_once()
        mock_data_module.assert_called_once()
        mock_trainer.assert_called_once_with(model=mock_model.return_value.to(), args=mock_training_args.return_value, data_collator=mock_data_collator.return_value, train_dataset=mock_data_module.return_value.train_dataset, eval_dataset=mock_data_module.return_value.val_dataset)

    def test_fine_tune(self):
        self.fine_tune_model = FineTune(self.config, self.tokenizer_name, self.model_name)
        self.fine_tune_model.config.path_to_data = re.sub(r"processed.csv", "test.csv", self.fine_tune_model.config.path_to_data)
        self.fine_tune_model.config.fp16 = False
        self.fine_tune_model.config.num_train_epochs = 0.001
        self.fine_tune_model.config.max_steps = 1
        self.fine_tune_model.config.gradient_accumulation_steps = 1
        self.fine_tune_model.config.path_to_save = self.fine_tune_model.config.path_to_save + '-test'
        self.fine_tune_model.prepare_model()
        self.fine_tune_model.fine_tune()
        assert os.path.exists(self.fine_tune_model.config.path_to_save)
        assert os.path.exists(self.fine_tune_model.config.path_to_save + '-test')

if __name__ == '__main__':
    unittest.main()
