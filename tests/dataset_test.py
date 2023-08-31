import pytest
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from src.data.dataset import (
    ApplyWordDropout,
    ConversationDataset,
    ConversationDataModule,
)


class TestConfig:
    def __init__(self):
        self.word_dropout = 0.2
        self.val_size = 0.2
        self.batch_size = 1
        self.df = pd.DataFrame(
            {
                "context": ["привет", "там хорошо, где нас нет."],
                "reply": ["пока", "мне в петлю лезть, а ей смешно"],
            }
        )


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(
        "tinkoff-ai/ruDialoGPT-medium", padding_side="left"
    )


@pytest.fixture
def word_dropout(tokenizer):
    return ApplyWordDropout(
        replace_with=tokenizer.convert_tokens_to_ids(tokenizer.unk_token),
        eos_token_id=tokenizer.eos_token_id,
        word_dropout=0.2,
    )


@pytest.fixture
def conversation_dataset():
    cfg = TestConfig()
    return ConversationDataset(cfg.df, cfg)


@pytest.fixture
def conversation_data_module():
    cfg = TestConfig()
    data_module = ConversationDataModule(cfg.df, cfg)
    data_module.setup()
    return data_module


def test_apply_word_dropout(word_dropout):
    input_tensor = torch.tensor([101, 2023, 2003, 1996, 2307, 102])
    dropout_tensor = input_tensor.clone()
    while torch.all(torch.eq(dropout_tensor, input_tensor)):
        dropout_tensor = word_dropout._apply_word_dropout(input_tensor)
    assert torch.any(torch.ne(dropout_tensor, input_tensor))


def test_concat_conv(conversation_dataset, tokenizer):
    sentences = ["Hello", "How are you?"]
    conv_flat = conversation_dataset._concat_conv(sentences, tokenizer)
    assert isinstance(conv_flat, torch.Tensor)
    assert torch.allclose(
        conv_flat,
        torch.tensor([50257, 44, 42911, 50258, 44, 1170, 2213, 9497, 35, 50257]),
    )


def test_train_dataloader(conversation_data_module):
    train_loader = conversation_data_module.train_dataloader()
    assert isinstance(train_loader, DataLoader)


def test_val_dataloader(conversation_data_module):
    val_loader = conversation_data_module.val_dataloader()
    assert isinstance(val_loader, DataLoader)
