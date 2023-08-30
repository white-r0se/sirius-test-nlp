import os
import pytest
import re
import transformers
from model.finetune import (
    Config,
    FineTune,
    ConversationDataModule,
    LoraConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@pytest.fixture
def fine_tune_model():
    config = Config()
    config.fp16 = False
    tokenizer_name = "tinkoff-ai/ruDialoGPT-medium"
    model_name = "tinkoff-ai/ruDialoGPT-medium"
    return FineTune(config, tokenizer_name, model_name)


def test_prepare_model(fine_tune_model):
    fine_tune_model.prepare_model()
    assert fine_tune_model.tokenizer is not None
    assert isinstance(
        fine_tune_model.tokenizer,
        transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast,
    )
    assert fine_tune_model.model is not None
    assert isinstance(
        fine_tune_model.model,
        transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel,
    )
    assert fine_tune_model.lora_config is not None
    assert isinstance(fine_tune_model.lora_config, LoraConfig)
    assert fine_tune_model.data_collator is not None
    assert isinstance(fine_tune_model.data_collator, DataCollatorForLanguageModeling)
    assert fine_tune_model.training_args is not None
    assert isinstance(fine_tune_model.training_args, TrainingArguments)
    assert fine_tune_model.data_module is not None
    assert isinstance(fine_tune_model.data_module, ConversationDataModule)
    assert fine_tune_model.trainer is not None
    assert isinstance(fine_tune_model.trainer, Trainer)


def test_fine_tune(fine_tune_model):
    fine_tune_model.config.path_to_data = re.sub(
        r"processed.csv", "test.csv", fine_tune_model.config.path_to_data
    )
    fine_tune_model.config.fp16 = False
    fine_tune_model.config.num_train_epochs = 0.001
    fine_tune_model.config.max_steps = 1
    fine_tune_model.config.gradient_accumulation_steps = 1
    fine_tune_model.config.path_to_save = fine_tune_model.config.path_to_save + "-test"
    fine_tune_model.prepare_model()
    fine_tune_model.fine_tune()
    assert os.path.exists(fine_tune_model.config.path_to_save)
