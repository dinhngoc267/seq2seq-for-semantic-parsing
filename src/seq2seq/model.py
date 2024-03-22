import torch
import copy
import pytorch_lightning as pl
from torchmetrics.text import BLEUScore
from src.metric import AccuracyMetric


class Encoder(pl.LightningModule):
    def __init__(self, vocab_size, embedding_layer, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.embedding = embedding_layer

        self.rnn = torch.nn.GRU(input_size=embedding_dim,
                                hidden_size=hidden_dim,
                                num_layers=num_layers,
                                dropout=dropout if num_layers > 1 else 0,
                                bidirectional=True,
                                batch_first=True)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        # output: [batch_size, seq_len, hidden_dim*2]; hidden: [2*n_layer, batch_size, hidden_dim]
        output, hidden = self.rnn(embedded)
        # concat forward and backward hidden state
        hidden = torch.cat((hidden[0::2, :, :], hidden[1::2, :, :]), dim=-1)
        # hidden = hidden[0:2,:,:]
        return output, hidden


class Attention(pl.LightningModule):
    def __init__(self, key_dim, query_dim, hidden_dim):
        super().__init__()

        self.K = torch.nn.Linear(key_dim, hidden_dim)
        self.Q = torch.nn.Linear(query_dim, hidden_dim)
        self.W = torch.nn.Linear(hidden_dim, 1)

    def forward(self, keys, query):
        # keys: [batch_size, seq_len, key_dim]
        K = self.K(keys)
        Q = self.Q(query)

        attention_scores = self.W(torch.nn.functional.tanh(K + Q))
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        context_vec = torch.bmm(attention_scores.permute(0, 2, 1), keys)

        return context_vec


class AttnDecoder(pl.LightningModule):
    def __init__(self, embedding_layer, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, input_seq_len,
                 sos_token_id):
        super().__init__()
        self.sos_token_id = sos_token_id
        self.vocab_size = vocab_size
        self.input_seq_len = input_seq_len

        self.attention = Attention(key_dim=2 * hidden_dim,
                                   query_dim=2 * hidden_dim,
                                   hidden_dim=hidden_dim)

        self.embedding = embedding_layer

        self.rnn = torch.nn.GRU(input_size=embedding_dim + hidden_dim * 2 + input_seq_len,
                                hidden_size=2 * hidden_dim,
                                num_layers=num_layers,
                                dropout=dropout if num_layers > 1 else 0,
                                bidirectional=False,
                                batch_first=True)

        self.generation_layer = torch.nn.Linear(in_features=2 * hidden_dim, out_features=vocab_size)
        self.dropout = torch.nn.Dropout(dropout)

    def one_step_forward(self,
                         decoder_input_ids,
                         encoder_outputs,
                         previous_decoder_hidden_state,
                         verbose=False):

        decoder_embedded = self.dropout(self.embedding(decoder_input_ids))
        # get the last layer decoder hidden states [n_layer, batch_size, hidden_dim] -> [1, batch_size, hidden_dim]
        query = previous_decoder_hidden_state[-1, :, :].unsqueeze(0)
        query = query.permute(1, 0, 2)  # [batch_size, 1, hidden_dim]

        if verbose:
            print(f'Query shape: {query.shape}')
            print(f'Keys shape: {encoder_outputs.shape}')

        # Compute attentive read base on attention mechanism
        attentive_read = self.attention(encoder_outputs, query)

        # Update decoder hidden state base on previous output, attentive read and previous selective read
        rnn_input = torch.cat((decoder_embedded, attentive_read), dim=-1)
        decoder_outputs, decoder_hidden_state = self.rnn(rnn_input, previous_decoder_hidden_state)

        # Compute Generation Score
        probs_g = self.generation_layer(decoder_outputs)
        probs = torch.nn.functional.log_softmax(probs_g, dim=-1)

        return probs, decoder_hidden_state

    def forward(self, encoder_outputs, encoder_hidden_state, max_len, ground_truth=None, output_ids=None,
                input_ids=None):
        batch_size = encoder_outputs.size(0)

        decoder_input_ids = torch.empty((batch_size, 1), device=self.device, dtype=torch.long).fill_(self.sos_token_id)
        decoder_hidden_state = encoder_hidden_state

        decoder_outputs = []
        for i in range(max_len):
            decoder_output, decoder_hidden_state = self.one_step_forward(decoder_input_ids,
                                                                         encoder_outputs,
                                                                         decoder_hidden_state,
                                                                         False)

            decoder_outputs.append(decoder_output)

            if ground_truth is not None:
                decoder_input_ids = output_ids[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                topi = topi.squeeze(-1)
                decoder_input_ids = topi.detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs


class Seq2SeqAttn(pl.LightningModule):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size,
                 num_layers,
                 dropout,
                 input_len,
                 sos_token_id,
                 vocab,
                 id2token,
                 max_output_length):
        super().__init__()

        self.vocab = vocab
        self.id2token = id2token
        self.max_output_length = max_output_length

        self.embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size,
                                                  embedding_dim=embedding_dim)

        self.encoder = Encoder(vocab_size=vocab_size,
                               embedding_dim=embedding_dim,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               embedding_layer=self.embedding_layer)

        self.decoder = AttnDecoder(vocab_size=vocab_size,
                                   embedding_dim=embedding_dim,
                                   hidden_dim=hidden_dim,
                                   num_layers=num_layers,
                                   input_seq_len=input_len,
                                   dropout=dropout,
                                   sos_token_id=sos_token_id,
                                   embedding_layer=self.embedding_layer)

        self.train_blue_score = BLEUScore()
        self.val_bleu_score = BLEUScore()
        self.train_accuracy = AccuracyMetric()
        self.val_accuracy = AccuracyMetric()

    def forward(self, input_ids, ground_truth=None, output_ids=None):

        encoder_outputs, encoder_hidden_state = self.encoder.forward(input_ids)

        decoder_outputs = self.decoder.forward(encoder_outputs,
                                               encoder_hidden_state,
                                               self.max_output_length,
                                               ground_truth,
                                               output_ids,
                                               input_ids)

        return decoder_outputs

    def loss(self, probs, ground_truth):
        return torch.nn.functional.cross_entropy(probs.view(-1, probs.shape[-1]), ground_truth.view(-1), ignore_index=0)

    def beam_search(self, predictions, k=3):
        batch_size, seq_length, vocab_size = predictions.shape
        log_prob, indices = predictions[:, 0, :].topk(k, sorted=True)
        indices = indices.unsqueeze(-1)
        for n1 in range(1, seq_length):
            log_prob_temp = log_prob.unsqueeze(-1) + predictions[:, n1, :].unsqueeze(1).repeat(1, k, 1)
            log_prob, index_temp = log_prob_temp.view(batch_size, -1).topk(k, sorted=True)
            idx_begin = index_temp // vocab_size
            idx_concat = index_temp % vocab_size
            new_indices = torch.zeros((batch_size, k, n1 + 1), device=self.device, dtype=torch.int64)
            for n2 in range(batch_size):
                new_indices[n2, :, :-1] = indices[n2][idx_begin[n2]]
                new_indices[n2, :, -1] = idx_concat[n2]
            indices = new_indices

        return indices, log_prob

    def decode_prediction(self, probs, input_tokens):
        # _, ids = probs.topk(1)
        ids, _ = self.beam_search(probs)
        # get the highest score
        ids = ids[:, 0, :]
        ids = ids.detach().cpu().numpy()

        tokens_buffer = []

        for row_idx, row in enumerate(ids):
            tmp = []
            for col_idx, id in enumerate(row):
                if id >= len(self.vocab):
                    tmp.append(input_tokens[row_idx].split(' ')[id - len(self.vocab)])
                else:
                    tmp.append(self.id2token[id])
            tmp = ' '.join([x for x in tmp if x != '<pad>']).split('</s>')[0].strip()
            tokens_buffer.append(tmp)
        return tokens_buffer

    def training_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        input_tokens = batch['input_tokens']
        output_ids = batch['output_ids']
        output_sequences = batch['output_sequences']
        labels = batch['labels']

        probs = self.forward(input_ids, labels, output_ids)
        loss = self.loss(probs, labels)

        # calculate blue_score
        predictions = self.decode_prediction(probs, input_tokens)

        accuracy = self.train_accuracy(predictions, output_sequences)
        bleu_score = self.train_blue_score(predictions, [[x] for x in output_sequences])

        batch_size = input_ids.size(0)
        self.log_dict(
            {'train_loss': loss,
             'train_bleu_score': bleu_score,
             'train_accuracy': accuracy
             },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )

        return loss

    def validation_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        input_tokens = batch['input_tokens']
        output_ids = batch['output_ids']
        output_sequences = batch['output_sequences']
        labels = batch['labels']

        probs = self.forward(input_ids, labels, output_ids)
        loss = self.loss(probs, labels)

        # calculate blue_score
        predictions = self.decode_prediction(probs, input_tokens)

        for idx in range(len(output_sequences)):
            if output_sequences[idx] != predictions[idx]:
                print("---------------------------------------")
                print(output_sequences[idx])
                print(predictions[idx])

        accuracy = self.val_accuracy(predictions, output_sequences)
        bleu_score = self.val_bleu_score(predictions, [[x] for x in output_sequences])

        batch_size = input_ids.size(0)
        self.log_dict(
            {'val_loss': loss,
             'val_bleu_score': bleu_score,
             'val_accuracy': accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

        return optimizer
