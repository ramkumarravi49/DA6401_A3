# import json
# from torch.utils.data import Dataset, DataLoader

# import torch
# import os
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn as nn


# # --------------------- Vocabulary Mapper ---------------------
# class VocabMapper:
#     def __init__(self, tokens=None, special_markers=['<pad>', '<sos>', '<eos>', '<unk>']):
#         self._special = special_markers
#         self._idx_to_char = list(special_markers) + (tokens or [])
#         self._char_to_idx = {c: i for i, c in enumerate(self._idx_to_char)}

#     @classmethod
#     def create_from_corpus(cls, samples):
#         charset = sorted(set(ch for text in samples for ch in text))
#         return cls(tokens=charset)

#     def persist(self, filename):
#         with open(filename, 'w', encoding='utf-8') as out_file:
#             json.dump(self._idx_to_char, out_file, ensure_ascii=False)

#     @classmethod
#     def load_from_file(cls, filename):
#         with open(filename, encoding='utf-8') as f:
#             idx_list = json.load(f)
#         obj = cls(tokens=[])
#         obj._idx_to_char = idx_list
#         obj._char_to_idx = {ch: i for i, ch in enumerate(idx_list)}
#         return obj

#     def convert_to_ids(self, string, sos=False, eos=False):
#         sos_token = [self._char_to_idx['<sos>']] if sos else []
#         eos_token = [self._char_to_idx['<eos>']] if eos else []
#         body = [self._char_to_idx.get(ch, self._char_to_idx['<unk>']) for ch in string]
#         return sos_token + body + eos_token

#     def convert_to_text(self, index_list, clean=True):
#         characters = map(lambda i: self._idx_to_char[i], index_list)
#         if clean:
#             return ''.join(filter(lambda c: c not in self._special, characters))
#         return ''.join(characters)

#     def get_pad_index(self): return self._char_to_idx['<pad>']
#     def get_sos_index(self): return self._char_to_idx['<sos>']
#     def get_eos_index(self): return self._char_to_idx['<eos>']
#     def get_unk_index(self): return self._char_to_idx['<unk>']
#     def total_vocab_size(self): return len(self._idx_to_char)

# # ------------------------ Data Utilities ------------------------
# def parse_dataset(split, lang_code):
#     # filepath = f"/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/{lang_code}/lexicons/{lang_code}.translit.sampled.{split}.tsv"
#     filepath = os.path.join("dakshina_dataset_v1.0", lang_code, "lexicons", f"{lang_code}.translit.sampled.{split}.tsv")

#     examples = []
#     with open(filepath, encoding='utf-8') as f:
#         for line in f:
#             cols = line.strip().split('\t')
#             if len(cols) >= 2:
#                 examples.append((cols[1], cols[0]))  # (src, tgt)
#     return examples


# def pad_and_batch(batch, src_vocab, tgt_vocab):
#     src_seqs, tgt_seqs = zip(*batch)
#     padded_src = pad_sequence(src_seqs, batch_first=True, padding_value=src_vocab.get_pad_index())
#     padded_tgt = pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_vocab.get_pad_index())
#     src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
#     return padded_src, src_lengths, padded_tgt




# class TransliterationDataset(Dataset):
#     def __init__(self, pairs, src_vocab, tgt_vocab):
#         self.data = []
#         for src, tgt in pairs:
#             tgt_ids = tgt_vocab.convert_to_ids(tgt, sos=True, eos=True)
#             src_ids = src_vocab.convert_to_ids(src, sos=True, eos=True)
            
#             self.data.append((torch.tensor(src_ids), torch.tensor(tgt_ids)))

#     def __getitem__(self, idx): return self.data[idx]

#     def __len__(self): return len(self.data)
    



# def prepare_dataloaders(language_code='hi', batch_sz=64, device='cpu',
#                         workers=2, prefetch=4, persistent=True):

#     combined = parse_dataset('train', language_code) + parse_dataset('dev', language_code)
#     src_vocab = VocabMapper.create_from_corpus([s for s, _ in combined])
#     tgt_vocab = VocabMapper.create_from_corpus([t for _, t in combined])

#     args = dict(batch_size=batch_sz,
#                 num_workers=workers,
#                 prefetch_factor=prefetch,
#                 persistent_workers=persistent,
#                 pin_memory=(device == 'cuda'))

#     def make_loader(split):
#         pairs = parse_dataset(split, language_code)
#         dataset = TransliterationDataset(pairs, src_vocab, tgt_vocab)
#         return DataLoader(dataset,
#                           shuffle=(split == 'train'),
#                           collate_fn=lambda batch: pad_and_batch(batch, src_vocab, tgt_vocab),
#                           **args)

#     return {s: make_loader(s) for s in ['train', 'dev', 'test']}, src_vocab, tgt_vocab

# #------------------------------------------------------------------------------------------------------
# class BahdanauAttention(nn.Module):
#     def __init__(self, enc_hid, dec_hid):
#         super().__init__()
#         self.attn = nn.Linear(enc_hid + dec_hid, dec_hid)
#         self.v = nn.Linear(dec_hid, 1, bias=False)

#     def forward(self, hidden, encoder_outputs, mask):
#         B, T, H = encoder_outputs.size()
#         hidden = hidden.unsqueeze(1).repeat(1, T, 1)
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
#         scores = self.v(energy).squeeze(2)
#         scores = scores.masked_fill(~mask, -1e9)
#         return torch.softmax(scores, dim=1)

# import json
# from torch.utils.data import Dataset, DataLoader

# import torch
# import os
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn as nn


# # --------------------- Vocabulary Mapper ---------------------
# class VocabMapper:
#     def __init__(self, tokens=None, special_markers=['<pad>', '<sos>', '<eos>', '<unk>']):
#         self._special = special_markers
#         self._idx_to_char = list(special_markers) + (tokens or [])
#         self._char_to_idx = {c: i for i, c in enumerate(self._idx_to_char)}

#     @classmethod
#     def create_from_corpus(cls, samples):
#         charset = sorted(set(ch for text in samples for ch in text))
#         return cls(tokens=charset)

#     def persist(self, filename):
#         with open(filename, 'w', encoding='utf-8') as out_file:
#             json.dump(self._idx_to_char, out_file, ensure_ascii=False)

#     @classmethod
#     def load_from_file(cls, filename):
#         with open(filename, encoding='utf-8') as f:
#             idx_list = json.load(f)
#         obj = cls(tokens=[])
#         obj._idx_to_char = idx_list
#         obj._char_to_idx = {ch: i for i, ch in enumerate(idx_list)}
#         return obj

#     def convert_to_ids(self, string, sos=False, eos=False):
#         sos_token = [self._char_to_idx['<sos>']] if sos else []
#         eos_token = [self._char_to_idx['<eos>']] if eos else []
#         body = [self._char_to_idx.get(ch, self._char_to_idx['<unk>']) for ch in string]
#         return sos_token + body + eos_token

#     def convert_to_text(self, index_list, clean=True):
#         characters = map(lambda i: self._idx_to_char[i], index_list)
#         if clean:
#             return ''.join(filter(lambda c: c not in self._special, characters))
#         return ''.join(characters)

#     def get_pad_index(self): return self._char_to_idx['<pad>']
#     def get_sos_index(self): return self._char_to_idx['<sos>']
#     def get_eos_index(self): return self._char_to_idx['<eos>']
#     def get_unk_index(self): return self._char_to_idx['<unk>']
#     def total_vocab_size(self): return len(self._idx_to_char)

# # ------------------------ Data Utilities ------------------------
# def parse_dataset(split, lang_code):
#     #filepath = f"/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/{lang_code}/lexicons/{lang_code}.translit.sampled.{split}.tsv"
#     filepath = os.path.join("dakshina_dataset_v1.0", lang_code, "lexicons", f"{lang_code}.translit.sampled.{split}.tsv")


#     examples = []
#     with open(filepath, encoding='utf-8') as f:
#         for line in f:
#             cols = line.strip().split('\t')
#             if len(cols) >= 2:
#                 examples.append((cols[1], cols[0]))  # (src, tgt)
#     return examples


# def pad_and_batch(batch, src_vocab, tgt_vocab):
#     src_seqs, tgt_seqs = zip(*batch)
#     padded_src = pad_sequence(src_seqs, batch_first=True, padding_value=src_vocab.get_pad_index())
#     padded_tgt = pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_vocab.get_pad_index())
#     src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
#     return padded_src, src_lengths, padded_tgt




# class TransliterationDataset(Dataset):
#     def __init__(self, pairs, src_vocab, tgt_vocab):
#         self.data = []
#         for src, tgt in pairs:
#             tgt_ids = tgt_vocab.convert_to_ids(tgt, sos=True, eos=True)
#             src_ids = src_vocab.convert_to_ids(src, sos=True, eos=True)
            
#             self.data.append((torch.tensor(src_ids), torch.tensor(tgt_ids)))

#     def __getitem__(self, idx): return self.data[idx]

#     def __len__(self): return len(self.data)
    



# def prepare_dataloaders(language_code='hi', batch_sz=64, device='cpu',
#                         workers=2, prefetch=4, persistent=True):

#     combined = parse_dataset('train', language_code) + parse_dataset('dev', language_code)
#     src_vocab = VocabMapper.create_from_corpus([s for s, _ in combined])
#     tgt_vocab = VocabMapper.create_from_corpus([t for _, t in combined])

#     args = dict(batch_size=batch_sz,
#                 num_workers=workers,
#                 prefetch_factor=prefetch,
#                 persistent_workers=persistent,
#                 pin_memory=(device == 'cuda'))

#     def make_loader(split):
#         pairs = parse_dataset(split, language_code)
#         dataset = TransliterationDataset(pairs, src_vocab, tgt_vocab)
#         return DataLoader(dataset,
#                           shuffle=(split == 'train'),
#                           collate_fn=lambda batch: pad_and_batch(batch, src_vocab, tgt_vocab),
#                           **args)

#     return {s: make_loader(s) for s in ['train', 'dev', 'test']}, src_vocab, tgt_vocab

# #------------------------------------------------------------------------------------------------------
# class BahdanauAttention(nn.Module):
#     def __init__(self, enc_hid, dec_hid):
#         super().__init__()
#         self.attn = nn.Linear(enc_hid + dec_hid, dec_hid)
#         self.v = nn.Linear(dec_hid, 1, bias=False)

#     def forward(self, hidden, encoder_outputs, mask):
#         B, T, H = encoder_outputs.size()
#         hidden = hidden.unsqueeze(1).repeat(1, T, 1)
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
#         scores = self.v(energy).squeeze(2)
#         scores = scores.masked_fill(~mask, -1e9)
#         return torch.softmax(scores, dim=1)

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import json



import torch.nn as nn

# ------------------------- Vocabulary Mapping -------------------------
class VocabMapper:
    def __init__(self, tokens=None, special_tokens=['<pad>', '<sos>', '<eos>', '<unk>']):
        self.special_tokens = special_tokens
        self.token_list = list(special_tokens) + (tokens or [])
        self.token_to_index = {char: idx for idx, char in enumerate(self.token_list)}

    @classmethod
    def create_from_corpus(cls, lines):
        unique_chars = sorted(set(char for line in lines for char in line))
        return cls(tokens=unique_chars)

    def encode_sequence(self, string, include_sos=False, include_eos=False):
        sequence = []
        if include_sos:
            sequence.append(self.token_to_index['<sos>'])
        sequence += [self.token_to_index.get(char, self.token_to_index['<unk>']) for char in string]
        if include_eos:
            sequence.append(self.token_to_index['<eos>'])
        return sequence

    def convert_to_text(self, indices, strip_specials=True):
        chars = [self.token_list[i] for i in indices]
        if strip_specials:
            chars = [c for c in chars if c not in self.special_tokens]
        return ''.join(chars)

    def get_pad_index(self):
        return self.token_to_index['<pad>']

    def get_sos_index(self):
        return self.token_to_index['<sos>']

    def get_eos_index(self):
        return self.token_to_index['<eos>']

    def get_unk_index(self):
        return self.token_to_index['<unk>']

    def total_vocab_size(self):
        return len(self.token_list)

# ------------------------- Data Reader -------------------------
def parse_dataset(split, lang_code):
    file_path = os.path.join("dakshina_dataset_v1.0", lang_code, "lexicons", f"{lang_code}.translit.sampled.{split}.tsv")
    pairs = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) >= 2:
                pairs.append((fields[1], fields[0]))  # (src, tgt)
    return pairs

# ------------------------- PyTorch Dataset -------------------------
class TransliterationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.data = []
        for src, tgt in pairs:
            src_encoded = src_vocab.encode_sequence(src, include_sos=True, include_eos=True)
            tgt_encoded = tgt_vocab.encode_sequence(tgt, include_sos=True, include_eos=True)
            self.data.append((torch.tensor(src_encoded), torch.tensor(tgt_encoded)))

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[index]

# ------------------------- Collation Function -------------------------
def pad_and_batch(batch, src_vocab, tgt_vocab):
    src_seqs, tgt_seqs = zip(*batch)
    padded_src = pad_sequence(src_seqs, batch_first=True, padding_value=src_vocab.get_pad_index())
    padded_tgt = pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_vocab.get_pad_index())
    src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    return padded_src, src_lengths, padded_tgt

class BatchCollator:
    def __init__(self, src_vocab, tgt_vocab):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __call__(self, batch):
        return pad_and_batch(batch, self.src_vocab, self.tgt_vocab)

# ------------------------- Dataloader Prep -------------------------
def prepare_dataloaders(language_code='hi', batch_sz=64, device='cpu',
                        workers=2, prefetch=4, persistent=True):
    combined = parse_dataset('train', language_code) + parse_dataset('dev', language_code)
    src_vocab = VocabMapper.create_from_corpus([s for s, _ in combined])
    tgt_vocab = VocabMapper.create_from_corpus([t for _, t in combined])
    args = dict(batch_size=batch_sz,
                num_workers=workers,
                prefetch_factor=prefetch,
                persistent_workers=persistent,
                pin_memory=(device == 'cuda'))

    collator = BatchCollator(src_vocab, tgt_vocab)

    def make_loader(split):
        pairs = parse_dataset(split, language_code)
        dataset = TransliterationDataset(pairs, src_vocab, tgt_vocab)
        return DataLoader(dataset, shuffle=(split == 'train'), collate_fn=collator, **args)

    loaders = {split: make_loader(split) for split in ['train', 'dev', 'test']}
    return loaders, src_vocab, tgt_vocab
#------------------------------------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear(enc_hid + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        B, T, H = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, T, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)
        scores = scores.masked_fill(~mask, -1e9)
        return torch.softmax(scores, dim=1)