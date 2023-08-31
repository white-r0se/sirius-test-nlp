import os
import pytest
import re
import transformers
import peft
from src.model.trainer import (
    Config,
    TrainerFineTune,
    ConversationDataModule,
    LoraConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@pytest.fixture
def fine_tune_model():
    with open("src/config/config.json", "r") as f:
        config = Config().model_validate_json(f.read())
    config.fp16 = False
    tokenizer_name = "tinkoff-ai/ruDialoGPT-medium"
    model_name = "tinkoff-ai/ruDialoGPT-medium"
    return TrainerFineTune(config, tokenizer_name, model_name)


def test_prepare_model(fine_tune_model):
    fine_tune_model.prepare_model()
    assert isinstance(
        fine_tune_model.tokenizer,
        transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast,
    )
    assert isinstance(
        fine_tune_model.model,
        peft.peft_model.PeftModelForCausalLM,
    )
    assert isinstance(fine_tune_model.lora_config, LoraConfig)
    assert isinstance(fine_tune_model.data_collator, DataCollatorForLanguageModeling)
    assert isinstance(fine_tune_model.training_args, TrainingArguments)
    assert isinstance(fine_tune_model.data_module, ConversationDataModule)
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
