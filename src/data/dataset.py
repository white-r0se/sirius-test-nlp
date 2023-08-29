import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from torch import Tensor


class ApplyWordDropout:
    """Apply word dropout to a tensor of token ids"""

    def __init__(self, replace_with: int, eos_token_id: int, word_dropout: float = 0.0):
        self.keep_prop = 1.0 - word_dropout
        self.replace_with = replace_with
        self.eos_token_id = eos_token_id

    def _apply_word_dropout(self, tensor: Tensor) -> Tensor:
        dropout_mask = torch.rand(tensor.shape) < self.keep_prop
        dropout_mask &= tensor != self.eos_token_id
        result = torch.where(
            dropout_mask, tensor, torch.full_like(tensor, self.replace_with)
        )
        return result

    def __call__(self, sample: Tensor) -> Tensor:
        return self._apply_word_dropout(sample)


class ConversationDataset(Dataset):
    """Dataset for conversation data"""

    def __init__(self, df: DataFrame, cfg):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "tinkoff-ai/ruDialoGPT-small", padding_side="left"
        )
        self.word_dropout = ApplyWordDropout(
            replace_with=self.tokenizer(self.tokenizer.unk_token)["input_ids"][0],
            eos_token_id=self.tokenizer.eos_token_id,
            word_dropout=cfg.word_dropout,
        )
        self.samples = []
        for _, sentences in df.iterrows():
            conv = self._concat_conv(sentences, self.tokenizer)
            self.samples.append(conv)
        if cfg.word_dropout:
            self.samples = [self.word_dropout(sample) for sample in self.samples]

    def _concat_conv(self, sentences, tokenizer):
        """Concatenate sentences into a single tensor"""
        eos_list = [50257, 50258, 50257, 50258, 50257]
        conv = [
            torch.cat(
                (
                    torch.tensor([eos_list.pop()]).unsqueeze(0),
                    tokenizer(sentence, return_tensors="pt")["input_ids"],
                ),
                dim=1,
            )
            for sentence in sentences
            if sentence != ""
        ]
        conv[-1] = torch.cat(
            (
                conv[-1],
                torch.tensor([eos_list.pop()]).unsqueeze(0),
            ),
            dim=1,
        )
        conv_flat = torch.cat(conv, dim=1).view(-1)
        # cut off the conv_flat if it is too long
        if conv_flat.shape[0] > 1000:
            conv_flat = conv_flat[-1000:]
        return conv_flat

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item].to(torch.long)


class ConversationDataModule(pl.LightningDataModule):
    """DataModule for conversation data"""

    def __init__(self, data, cfg):
        super().__init__()
        train_data, val_data = train_test_split(data, test_size=cfg.val_size)
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_dataset = ConversationDataset(self.train_data, self.cfg)
        self.val_dataset = ConversationDataset(self.val_data, self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg.batch_size, collate_fn=self._collate
        )

    def _collate(self, examples: list[torch.Tensor]):
        max_length = max([len(ex) for ex in examples])
        padded_examples = [F.pad(ex, (max_length - len(ex), 0)) for ex in examples]
        return torch.stack(padded_examples, dim=0)
