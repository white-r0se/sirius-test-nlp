import unittest
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from ..data.dataset import ApplyWordDropout, ConversationDataset, ConversationDataModule


class MockConfig:
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


class TestApplyWordDropout(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "tinkoff-ai/ruDialoGPT-small", padding_side="left"
        )
        self.word_dropout = ApplyWordDropout(
            replace_with=self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token),
            eos_token_id=self.tokenizer.eos_token_id,
            word_dropout=0.2,
        )

    def test_apply_word_dropout(self):
        input_tensor = torch.tensor([101, 2023, 2003, 1996, 2307, 102])
        dropout_tensor = input_tensor.clone()
        while torch.all(torch.eq(dropout_tensor, input_tensor)):
            dropout_tensor = self.word_dropout._apply_word_dropout(input_tensor)
        self.assertTrue(torch.any(torch.ne(dropout_tensor, input_tensor)))


class TestConversationDataset(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        self.dataset = ConversationDataset(self.cfg.df, self.cfg)

    def test_concat_conv(self):
        sentences = ["Hello", "How are you?"]
        tokenizer = AutoTokenizer.from_pretrained(
            "tinkoff-ai/ruDialoGPT-small", padding_side="left"
        )
        conv_flat = self.dataset._concat_conv(sentences, tokenizer)
        self.assertTrue(isinstance(conv_flat, torch.Tensor))
        self.assertTrue(
            torch.all(
                torch.eq(
                    conv_flat,
                    torch.tensor(
                        [50257, 44, 42911, 50258, 44, 1170, 2213, 9497, 35, 50257]
                    ),
                )
            )
        )


class TestConversationDataModule(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        self.data_module = ConversationDataModule(self.cfg.df, self.cfg)
        self.data_module.setup()

    def test_train_dataloader(self):
        train_loader = self.data_module.train_dataloader()
        self.assertIsNotNone(train_loader)
        self.assertTrue(isinstance(train_loader, DataLoader))

    def test_val_dataloader(self):
        val_loader = self.data_module.val_dataloader()
        self.assertIsNotNone(val_loader)
        self.assertTrue(isinstance(val_loader, DataLoader))


if __name__ == "__main__":
    unittest.main()
