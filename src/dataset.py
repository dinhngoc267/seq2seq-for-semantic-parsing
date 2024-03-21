import pandas as pd
import copy
import torch
from utils import tokenize_sequence
from torch.utils.data import DataLoader, Dataset


class ParserDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 max_input_length,
                 max_output_length,
                 vocab,
                 device):
        super().__init__()
        self.device = device

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.data = data

    def __len__(self):
        return len(self.data)

    def encode_sequence(self, sequence, max_length):
        tokens = tokenize_sequence(sequence)

        ids = [self.vocab[token] if token in self.vocab else self.vocab['<unk>'] for token in tokens] + [self.vocab['</s>']]
        ids = (ids[:max_length] + [self.vocab['<pad>']] * (max_length - len(ids)))

        return tokens, ids

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        input_text = item['inputs']
        output_text = item['outputs']

        input_tokens, input_ids = self.encode_sequence(input_text, self.max_input_length)
        input_tokens = input_tokens[:self.max_input_length - 1] + ['</s>'] + ['<pad>'] * (self.max_input_length - 1 - len(input_tokens))

        output_tokens = tokenize_sequence(output_text)
        output_ids = [self.vocab[token] if token in self.vocab else self.vocab['<unk>'] for token in output_tokens] + [self.vocab['</s>']]
        output_ids = output_ids[:self.max_output_length] + [self.vocab['<pad>']] * (self.max_output_length - len(output_ids))

        output_sequences = ' '.join(output_tokens).lower()

        labels = copy.deepcopy(output_ids)
        occurrence = {}

        for idx, token in enumerate(output_tokens):
            if token in input_tokens:
                if token not in occurrence:
                    pos_in_input = [i for i, x in enumerate(input_tokens) if x == token]
                    occurrence[token] = [0, pos_in_input]
                else:
                    if occurrence[token][0] + 1 < len(occurrence[token][1]):
                        occurrence[token][0] += 1
                labels[idx] = self.vocab_size + occurrence[token][1][occurrence[token][0]]

        input_tokens = ' '.join(input_tokens)

        return dict(
            input_text=input_text,
            input_tokens=input_tokens,
            output_text=output_text,
            input_ids=torch.tensor(input_ids),
            output_ids=torch.tensor(output_ids),
            output_sequences=output_sequences,
            labels=torch.tensor(labels)
        )
