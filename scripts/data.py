# scripts/data.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from typing import Tuple


SPECIALS = ['<unk>', '<pad>']


def _build_vocab() -> Tuple[Vocab, callable]:
    tok = get_tokenizer("basic_english")
    counter = Counter()
    for line in WikiText2(split='train'):
        counter.update(tok(line))
    vocab = Vocab(counter, specials=SPECIALS)
    vocab.set_default_index(vocab['<unk>'])
    return vocab, tok


def _encode_split(vocab: Vocab, tok, split: str) -> torch.Tensor:
    ids = []
    for line in WikiText2(split=split):
        ids.extend([vocab[token] for token in tok(line)])
    # add simple EOS between lines to avoid cross-line leakage (optional)
    return torch.tensor(ids, dtype=torch.long)


class LMChunkDataset(Dataset):
    """
    Turn a 1D token-id stream into fixed-length [T] inputs and [T] targets.
    No padding; last partial chunk is dropped (keeps shapes consistent).
    """
    def __init__(self, ids: torch.Tensor, seq_len: int):
        assert ids.dim() == 1
        self.ids = ids
        self.seq_len = int(seq_len)
        # we need at least (seq_len + 1) tokens to form one (x, y) pair
        self.n = (len(ids) - 1) // self.seq_len

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx: int):
        i = idx * self.seq_len
        x = self.ids[i: i + self.seq_len]                 # [T]
        y = self.ids[i + 1: i + 1 + self.seq_len]         # [T]
        return x, y


def _collate(batch):
    # All items are length-T already; just stack to [B, T]
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def get_wikitext2_data(batch_size: int = 16,
                       seq_len: int = 256,
                       debug: bool = False):
    """
    Returns:
      train_loader, val_loader, test_loader, vocab
    where each loader yields (inputs [B,T], targets [B,T]).
    """
    vocab, tok = _build_vocab()
    pad_id = vocab['<pad>']  # kept for completeness; we don't pad in this pipeline

    train_ids = _encode_split(vocab, tok, 'train')
    val_ids   = _encode_split(vocab, tok, 'valid')
    test_ids  = _encode_split(vocab, tok, 'test')

    if debug:
        train_ids = train_ids[:50_000]   # tiny debug slice
        val_ids   = val_ids[:10_000]
        test_ids  = test_ids[:10_000]

    train_ds = LMChunkDataset(train_ids, seq_len)
    val_ds   = LMChunkDataset(val_ids,   seq_len)
    test_ds  = LMChunkDataset(test_ids,  seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0, collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=True, num_workers=0, collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              drop_last=True, num_workers=0, collate_fn=_collate)

    # Attach a couple of helpful attributes expected by downstream code.
    # (len(vocab) works for vocab size; pad_token_id provided for CE ignore_index if needed)
    vocab.pad_token_id = pad_id

    return train_loader, val_loader, test_loader, vocab
