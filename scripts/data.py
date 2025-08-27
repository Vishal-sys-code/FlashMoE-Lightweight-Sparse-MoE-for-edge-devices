import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

class Wikitext2Dataset(Dataset):
    """Custom PyTorch Dataset for Wikitext-2."""
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        # The number of sequences we can create
        return (self.data.size(0) - 1) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        # Input is the sequence, target is the sequence shifted by one
        inputs = self.data[start_idx:end_idx]
        targets = self.data[start_idx + 1:end_idx + 1]
        return inputs, targets

def get_wikitext2_data(batch_size: int, seq_len: int, cache_dir: str = "./artifacts/data", debug: bool = False):
    """
    Loads, tokenizes, and prepares the Wikitext-2 dataset for language modeling.

    Args:
        debug (bool): If True, uses a small subset of the data for fast testing.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader, tokenizer).
    """
    logging.info("Loading GPT2 tokenizer...")
    # Using GPT2 tokenizer as a standard subword tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Loading Wikitext-2 dataset...")
    try:
        # Try to load from cache first
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=cache_dir)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}. Ensure you have an internet connection.")
        raise

    def tokenize_function(examples):
        # Concatenate texts to handle documents properly
        return tokenizer(examples['text'], add_special_tokens=True, truncation=False)

    logging.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def process_split(split_name):
        # Concatenate all tokens from the split into a single tensor
        all_tokens = [tok for sample in tokenized_datasets[split_name] for tok in sample['input_ids']]
        return torch.tensor(all_tokens, dtype=torch.long)

    train_data = process_split('train')
    val_data = process_split('validation')
    test_data = process_split('test')

    if debug:
        logging.warning("Debug mode is enabled. Using a small subset of the data.")
        num_debug_batches = 10
        train_data = train_data[:batch_size * seq_len * num_debug_batches]
        val_data = val_data[:batch_size * seq_len * num_debug_batches]
        test_data = test_data[:batch_size * seq_len * num_debug_batches]
    
    logging.info(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}, Test tokens: {len(test_data)}")

    train_dataset = Wikitext2Dataset(train_data, seq_len)
    val_dataset = Wikitext2Dataset(val_data, seq_len)
    test_dataset = Wikitext2Dataset(test_data, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    logging.info(f"Created DataLoaders with {len(train_loader)} train batches, {len(val_loader)} val batches.")
    
    return train_loader, val_loader, test_loader, tokenizer

if __name__ == '__main__':
    # Example of how to use the data loader
    logging.basicConfig(level=logging.INFO)
    B, T = 16, 256
    train_loader, val_loader, test_loader, tokenizer = get_wikitext2_data(batch_size=B, seq_len=T)
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Fetch one batch
    inputs, targets = next(iter(train_loader))
    print(f"Input batch shape: {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")
    
    # Decode an example
    print("\n--- Example Decoded ---")
    print("Input: ", tokenizer.decode(inputs[0].tolist()))
    print("Target:", tokenizer.decode(targets[0].tolist()))
    print("-----------------------")
