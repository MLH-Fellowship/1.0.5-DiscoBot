import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(
        self, input_dim, emb_dim, hid_dim, output_dim, n_layer, dropout, pad_idx
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layer, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(2 * hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):
        # [seq_len, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(text))
        # https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, enforce_sorted=False
        )
        packed_out, (hidden, _) = self.lstm(packed_emb)

        # outputs : [seq_len, batch_size, n_direction * hid_dim]
        # hid : [n_layers * n_direction, batch_size, hid_dim]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

        # [batch_size, hid_dim]
        hidden_fwd, hidden_bck = hidden[-2], hidden[-1]
        # [batch_size, hid_dim*2]
        hidden = torch.cat((hidden_fwd, hidden_bck), dim=1)
        # pred : [batch_size, output_dim]
        return self.fc(self.dropout(hidden))
