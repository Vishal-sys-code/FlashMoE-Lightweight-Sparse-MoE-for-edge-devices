# scripts/data.py
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


class LMBlockDataset(Dataset):
    """
    Takes a long tokenized sequence and cuts it into fixed-length blocks.
    """
    def __init__(self, token_ids, block_size):
        self.block_size = block_size
        n_blocks = (len(token_ids) - 1) // block_size
        self.inputs = []
        self.targets = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            self.inputs.append(torch.tensor(token_ids[start:end], dtype=torch.long))
            self.targets.append(torch.tensor(token_ids[start+1:end+1], dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def get_wikitext2_data(batch_size=16, seq_len=128, debug=False):
    """
    Returns train/val/test DataLoaders and the tokenizer.
    Uses HuggingFace datasets + AutoTokenizer (GPT2 tokenizer by default).
    """
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Use GPT2 tokenizer (or switch to another if you want)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=False)["input_ids"]

    train_ids = sum(dataset["train"]["text"], [])
    val_ids = sum(dataset["validation"]["text"], [])
    test_ids = sum(dataset["test"]["text"], [])

    # Tokenize (flatten text into one big list of IDs)
    train_tokens = tokenizer(" ".join(train_ids), return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze().tolist()
    val_tokens = tokenizer(" ".join(val_ids), return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze().tolist()
    test_tokens = tokenizer(" ".join(test_ids), return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze().tolist()

    # Wrap into block dataset
    train_ds = LMBlockDataset(train_tokens, seq_len)
    val_ds = LMBlockDataset(val_tokens, seq_len)
    test_ds = LMBlockDataset(test_tokens, seq_len)

    # Subset for debug mode
    if debug:
        train_ds = torch.utils.data.Subset(train_ds, range(200))
        val_ds = torch.utils.data.Subset(val_ds, range(50))
        test_ds = torch.utils.data.Subset(test_ds, range(50))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, tokenizer
