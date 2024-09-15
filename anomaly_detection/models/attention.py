import torch
from torch import nn
import os
import math


class LearnablePosEnc(nn.Module):
    def __init__(self, series_len, embedding_dim, dropout=0.1):
        super(LearnablePosEnc, self).__init__()
        self.pos_enc = nn.Parameter(torch.empty(series_len, 1, embedding_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.pos_enc, -0.02, 0.02)

    def forward(self, x):
        '''
        x: (seq_len, batchs, channels)
        out: (seq_len, batchs, channels)
        '''
        return self.dropout(x + self.pos_enc)


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py  # noqa
class FixedPosEnc(nn.Module):
    """Inject some information about the relative or absolute position of the
        tokens in the sequence. The positional encodings have the same
        dimension as the embeddings, so that the two can be summed. Here, we
        use sine and cosine functions of different frequencies.

    Args:
        series_len: the max. length of the incoming sequence
        embedding_dim: the embed dim (required).
        dropout: the dropout value (default=0.1).
    """

    def __init__(self, series_len, embedding_dim,
                 dropout=0.1, scale_factor=1.0):
        super(FixedPosEnc, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(series_len, embedding_dim)  # positional encoding
        position = torch.arange(0, series_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float()
                             * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        # this stores the variable in the state_dict (used for non-trainable
        # variables)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_norm(is_batch):
    if is_batch:
        return nn.BatchNorm1d
    return nn.LayerNorm


def get_pos_enc(learnable):
    if learnable:
        return LearnablePosEnc
    return FixedPosEnc


class EncoderMTSC(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, dim_ff=128):
        super(EncoderMTSC, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, dim_ff)
        self.dropout2 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(dim_ff, d_model)
        self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.activation = nn.GELU()

    def forward(self, src, src_mask, src_key_padding_mask):
        '''
        input: word embeddings with position info. (seq_len, batch, channels)

        - fulfills interface for pytorch TransformEncoderLayer -
        '''

        ''' self attention '''
        # need to be of the format query key value
        x_attention_score = self.self_attn(src, src, src)[0]

        ''' residual connection '''
        # add in residual connection (seq_len, batch_size, d_model)
        src = src + self.dropout1(x_attention_score)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)

        ''' feed forward network '''
        src2 = self.fc2(self.dropout(self.activation(self.fc1(src))))

        '''residual connection'''
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)

        return src


class attentionMTSC(nn.Module):
    def __init__(self, series_len, input_dim, learnable_pos_enc=True,
                 d_model=128, heads=8, classes=6, dropout=0.1, dim_ff=256,
                 num_layers=1, task="classification", weights_fp=""):
        super(attentionMTSC, self).__init__()

        self.d_model = d_model
        self.tok_embed = nn.Linear(input_dim, d_model)

        self.pos_enc = get_pos_enc(
            learnable_pos_enc)(series_len, d_model, dropout=dropout)
        self.task = task

        encoder_layer = EncoderMTSC(d_model=d_model,
                                    heads=heads,
                                    dim_ff=dim_ff,
                                    dropout=dropout)

        # instead of using predefined TransformerEncoderLayer, user-defined
        # class allows us to specify batch or layer normalization
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(d_model, input_dim)

        self.activation = nn.GELU()

        if len(weights_fp):
            self.load_pretrained(weights_fp)

        if task == "classification":
            self.output = nn.Linear(series_len * d_model, classes)
            torch.nn.init.uniform_(self.output.weight, -0.02, 0.02)

    def load_pretrained(self, weights_fp):
        assert os.path.exists(weights_fp), \
            "No pre-trained state dictionary available"
        pre_trained_state_dict = torch.load(weights_fp)

        self.load_state_dict(pre_trained_state_dict)
        print(f"Pre-trained weights from {weights_fp} are loaded successfully")

    def forward(self, X):  # assume X is scaled
        '''
        input: X - (batch, seq_len, channels)
        '''
        # permute because pytorch convention for transformers is
        # [seq_length, batch_size, feat_dim]
        X = X.permute(1, 0, 2)

        '''   project into model dim space (embeddings)   '''
        # project input vectors to d_model dimensional space
        # [seq_length, batch_size, d_model]
        x_embed = self.tok_embed(X) * math.sqrt(self.d_model)

        '''   positional encoding   '''
        # add in positional information
        x_pos_enc = self.pos_enc(x_embed)

        '''   attention module   '''
        # (seq_len, batch_size, d_model)
        attention_output = self.encoder(x_pos_enc)
        attention_output = self.activation(attention_output)

        # (batch_size, seq_len, d_model)
        attention_output = attention_output.permute(1, 0, 2)
        attention_output = self.dropout(attention_output)

        '''   output layer   '''
        if self.task == "classification":
            # eliminates one dimension
            out = self.output(
                attention_output.reshape(attention_output.shape[0], -1))
        else:
            # (batch_size, seq_len, feat_dim)
            out = self.output(attention_output)

        return out
