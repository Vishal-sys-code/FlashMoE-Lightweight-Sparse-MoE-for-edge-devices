import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast

def load_wikitext2(batch_size, seq_len=256):
    """
    Loads and preprocesses the Wikitext-2 dataset.

    Args:
        batch_size (int): The batch size for the data loaders.
        seq_len (int): The sequence length for each sample.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): The data loader for the training set.
            - val_loader (DataLoader): The data loader for the validation set.
            - test_loader (DataLoader): The data loader for the test set.
            - vocab_size (int): The size of the vocabulary.
    """
    # for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Filter out empty lines
    dataset = dataset.filter(lambda example: len(example['text'].strip()) > 0)

    # Train the tokenizer
    tokenizer_path = "artifacts/wikitext2/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            dataset['train']['text'],
            vocab_size=30522,
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        tokenizer.save(tokenizer_path)
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})


    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False, padding=False, return_attention_mask=False)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])

    # Concatenate all tokens and create sequences
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // seq_len) * seq_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=1,
    )

    class Wikitext2Dataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            inputs = torch.tensor(item['input_ids'])
            labels = inputs.clone()
            return inputs, labels

    train_dataset = Wikitext2Dataset(lm_datasets["train"])
    val_dataset = Wikitext2Dataset(lm_datasets["validation"])
    test_dataset = Wikitext2Dataset(lm_datasets["test"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = tokenizer.vocab_size

    return train_loader, val_loader, test_loader, vocab_size

if __name__ == '__main__':
    train_loader, val_loader, test_loader, vocab_size = load_wikitext2(batch_size=8)
    print(f"Vocab size: {vocab_size}")
    print(f"Train loader has {len(train_loader)} batches")
    print(f"Val loader has {len(val_loader)} batches")
    print(f"Test loader has {len(test_loader)} batches")
    
    # Check a batch
    x, y = next(iter(train_loader))
    print(f"Batch shape: x: {x.shape}, y: {y.shape}")
    assert x.shape == y.shape
    assert x.shape[0] == 8
    assert x.shape[1] == 256
    print("Test block passed!")