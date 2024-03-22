import pandas as pd
import copy
import torch
import re
from torch.utils.data import Dataset
from src.utils import tokenize_sequence, compact_query, create_vocab
from typing import Optional


class SeqSeqAttnParserDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 device: torch.device,
                 max_input_length: Optional[int] = None,
                 max_output_length: Optional[int] = None,
                 vocab: Optional[dict] = None,
                 keywords_path: Optional[str] = None):
        """
        :param data_path: path of data which is csv format
        :param vocab: vocab which keys are words and values are ids
        :param device: cuda or cpu
        :param max_input_length: max length of input tokens which is None by default
        :param max_output_length: max length of output tokens which is None by default
        """

        super().__init__()

        self.device = device

        data = pd.read_csv(data_path)
        data.drop_duplicates(inplace=True)

        questions = [x.strip().replace("–", "-").replace('\n', ' ').replace('\r', '').replace('\t', '')
                     for x in data['Questions'].dropna().tolist()]
        queries = [x.strip().replace("–", "-").replace('\n', ' ').replace('\r', '').replace('\t', '')
                   for x in data['Queries'].dropna().tolist()]

        quoted = re.compile('"[^"]*"')
        entities = []
        for idx, q in enumerate(queries):
            for value in quoted.findall(q):
                value = value.replace('"', '')
                entities.append(value)

        queries = [compact_query(x) for x in queries]

        self.data = pd.DataFrame(data=zip(questions, queries), columns=['inputs', 'outputs'])

        if max_input_length is None:
            self.max_input_length = max([len(tokenize_sequence(item)) for item in questions]) + 2
            self.max_output_length = max([len(tokenize_sequence(item)) for item in queries]) + 3
        else:
            self.max_input_length = max_input_length
            self.max_output_length = max_output_length

        print(f"Train Dataset size: {len(data)}")

        if vocab is None:
            self.vocab = create_vocab(train_questions=questions,
                                      train_queries=queries,
                                      keywords_path=keywords_path)
        else:
            self.vocab = vocab
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {len(self.vocab)}")

        self.id2word = {}
        for token, token_id in self.vocab.items():
            self.id2word[token_id] = token

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab

    def get_id2word_mapping(self):
        return self.id2word

    def get_max_input_length(self):
        return self.max_input_length

    def get_max_output_length(self):
        return self.max_output_length

    def encode_sequence(self, sequence, max_length):
        tokens = tokenize_sequence(sequence)

        ids = [self.vocab[token] if token in self.vocab else self.vocab['<unk>']
               for token in tokens] + [self.vocab['</s>']]
        ids = ids[:max_length] + [self.vocab['<pad>']] * (max_length - len(ids))

        return tokens, ids

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        input_text = item['inputs']
        output_text = item['outputs']

        input_tokens, input_ids = self.encode_sequence(input_text, self.max_input_length)
        input_tokens = input_tokens[:self.max_input_length - 1] + ['</s>'] + ['<pad>'] * (
                    self.max_input_length - 1 - len(input_tokens))

        output_tokens = tokenize_sequence(output_text)
        output_ids = [self.vocab[token] if token in self.vocab else self.vocab['<unk>']
                      for token in output_tokens] + [self.vocab['</s>']]
        output_ids = output_ids[:self.max_output_length] + [self.vocab['<pad>']] * (
                    self.max_output_length - len(output_ids))

        output_sequences = ' '.join(output_tokens).lower()
        input_tokens = ' '.join(input_tokens)

        return dict(
            input_text=input_text,
            input_tokens=input_tokens,
            output_text=output_text,
            input_ids=torch.tensor(input_ids),
            output_ids=torch.tensor(output_ids),
            output_sequences=output_sequences
        )
