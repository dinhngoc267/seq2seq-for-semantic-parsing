import torch
import copy
import pytorch_lightning as pl
from fastNLP import Accuracy
from torchmetrics.text import BLEUScore
from metric import AccuracyMetric

class Encoder(pl.LightningModule):
    def __init__(self, vocab_size, embedding_layer, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        # self.embedding =  torch.nn.Embedding(num_embeddings=vocab_size,
        #                                      embedding_dim=embedding_dim)

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


class Decoder(pl.LightningModule):
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

        # self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
        #                                     embedding_dim=embedding_dim)

        self.rnn = torch.nn.GRU(input_size=embedding_dim + hidden_dim * 2 + input_seq_len,
                                hidden_size=2 * hidden_dim,
                                num_layers=num_layers,
                                dropout=dropout if num_layers > 1 else 0,
                                bidirectional=False,
                                batch_first=True)

        self.generation_layer = torch.nn.Linear(in_features=2 * hidden_dim, out_features=vocab_size)
        self.copy_W = torch.nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim * 2)
        self.copy_layer = torch.nn.Linear(in_features=self.input_seq_len, out_features=input_seq_len)
        self.dropout = torch.nn.Dropout(dropout)

    def one_step_forward(self, decoder_input_ids, encoder_outputs, previous_decoder_hidden_state,
                         previous_selective_read, encoder_input_ids, verbose=False):
        batch_size = encoder_outputs.size(0)

        decoder_embedded = self.dropout(self.embedding(decoder_input_ids))
        # get the last layer decoder hidden states [n_layer, batch_size, hidden_dim] -> [1, batch_size, hidden_dim]
        query = previous_decoder_hidden_state[-1, :, :].unsqueeze(0)
        query = query.permute(1, 0, 2)  # [batch_size, 1, hidden_dim]

        if verbose:
            print(f'Query shape: {query.shape}')
            print(f'Keys shape: {encoder_outputs.shape}')

        # Compute attentive read base on attention mechanism
        attentive_read = self.attention(encoder_outputs, query)

        # Update decoder hidden state base on previous output, attentive read and previous selection read
        rnn_input = torch.cat((decoder_embedded, attentive_read, previous_selective_read), dim=-1)
        decoder_outputs, hidden_state = self.rnn(rnn_input, previous_decoder_hidden_state)

        # Compute Generation Score
        probs_g = self.generation_layer(decoder_outputs)
        # probs_g = torch.nn.functional.relu(probs_g)
        # print(torch.max(probs_g))
        if verbose:
            print(f'Generation scores shape: {probs_g.shape}')

        # Compute Copy Score
        probs_c = self.copy_W(encoder_outputs)
        # apply non-linear activation to copy scores
        if verbose:
            print(f'Copy scores shape: {probs_c.shape}')
            print(f'Encoder outputs shape: {encoder_outputs.shape}')
        # probs_c = torch.bmm(probs_c, encoder_outputs.permute(0,2,1)) #, copy_scores.permute(0,2,1))
        probs_c = torch.bmm(decoder_outputs, probs_c.permute(0, 2, 1))
        # probs_c = self.copy_layer(probs_c)
        if verbose:
            print(f'Copy scores shape: {probs_c.shape}')
            # batch_size, 1, source_seq_len
        # Combine two scores
        # 1. get probs of source tokens in the probs_g
        probs_c = probs_c.squeeze(1)
        probs_g = probs_g.squeeze(1)

        source_tokens_probs = probs_g[torch.arange(probs_g.size(0)).unsqueeze(1), encoder_input_ids]  # batch_size, input_seq_len
        # I dont add probs_g of special tokens to probs_c
        mask = copy.deepcopy(encoder_input_ids)
        mask[(mask == 0) | (mask == 1) | (mask == 2)] = 0
        mask[mask != 0] = 1
        source_tokens_probs = torch.mul(mask, source_tokens_probs)  # mask*source_tokens_probs
        # add source_tokens_probs to copy prob
        combined_probs = source_tokens_probs + probs_c
        probs = torch.zeros((batch_size, self.vocab_size + self.input_seq_len), device=self.device)

        probs[:, self.vocab_size:] = combined_probs  # combined_probs #probs_c
        probs[:, :self.vocab_size] = probs_g

        probs = torch.nn.functional.log_softmax(probs, dim=-1)
        probs = probs.unsqueeze(1)
        probs_c = probs_c.unsqueeze(1)

        if verbose:
            print(f'Probs shape: {probs.shape}')

        # Compute selection read
        # get the token with the highest prob which is the "selection"
        _, topi = probs.topk(1)
        topi = topi.squeeze(-1).flatten()  # (batch_size, 1)

        # convert to one hot matrix [batch_size, vocab + source_seq_len]
        mask = torch.nn.functional.one_hot(topi, self.vocab_size + self.input_seq_len)
        # only get token from source [batch_size, source_seq_len]
        mask = mask[:, self.vocab_size:].unsqueeze(1)
        selection_read = mask * probs_c  # .unsqueeze(1)
        selection_read = selection_read / (torch.sum(selection_read, dim=-1, keepdim=True) + 1e-10)  # batch_size, 1, source_seq_len
        # selection_read = torch.bmm(selection_read, encoder_outputs)
        return probs, hidden_state, selection_read

    def forward(self, encoder_outputs, encoder_hidden_state, max_len, ground_truth=None, output_ids=None,
                input_ids=None):
        batch_size = encoder_outputs.size(0)

        decoder_input_ids = torch.empty((batch_size, 1), device=self.device, dtype=torch.long).fill_(self.sos_token_id)
        decoder_hidden_state = encoder_hidden_state
        selection_read = torch.zeros((batch_size, 1, input_ids.size(-1)), device=self.device)

        decoder_outputs = []
        for i in range(max_len):
            decoder_output, decoder_hidden_state, selection_read = self.one_step_forward(decoder_input_ids,
                                                                                         encoder_outputs,
                                                                                         decoder_hidden_state,
                                                                                         selection_read, input_ids,
                                                                                         False)
            decoder_outputs.append(decoder_output)
            if ground_truth is not None:
                decoder_input_ids = output_ids[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                topi = topi.squeeze(-1)
                # if the output_id > vocab_size it means the predicted token is taken from source sentence.
                for k in range(len(topi)):  # loop through batch
                    if topi[k][0] >= self.vocab_size:
                        token_idx_in_source = topi[k][0] - self.vocab_size
                        topi[k][0] = input_ids[k][token_idx_in_source]
                decoder_input_ids = topi.detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs


class CopySeq2Seq(pl.LightningModule):
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

        self.decoder = Decoder(vocab_size=vocab_size,
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

        # self.precision_score = Precision()

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
        acc = self.train_accuracy(predictions, output_sequences)
        output_sequences = [[x] for x in output_sequences]
        bleu_score = self.train_blue_score(predictions, output_sequences)
        batch_size = input_ids.size(0)
        self.log_dict(
            {'train_loss': loss, 'train_bleu_score': bleu_score, 'train_accuracy': acc}, #'train_precision_score': precision_score,
             # 'train_recall_score': recall_score},
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

        acc = self.val_accuracy(predictions, output_sequences)

        output_sequences = [[x] for x in output_sequences]
        bleu_score = self.val_bleu_score(predictions, output_sequences)

        batch_size = input_ids.size(0)
        self.log_dict(
            {'val_loss': loss,
             'val_bleu_score': bleu_score,
             'val_accuracy': acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)

        return optimizer
