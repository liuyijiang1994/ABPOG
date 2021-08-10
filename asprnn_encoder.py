import logging
import torch
import torch.nn as nn
import math
import logging
from aspGRU import BiAspectGRULayer
from torch.nn.utils.rnn import invert_permutation


class RNNEncoder(nn.Module):
    """
    Base class for rnn encoder
    """

    def forward(self, src, src_lens, src_mask=None, title=None, title_lens=None, title_mask=None):
        raise NotImplementedError


class RNNEncoderBasic(RNNEncoder):
    def __init__(self, vocab_size, embed_size, asp_size, hidden_size, num_layers, bidirectional, pad_token,
                 dropout=0.0):
        super(RNNEncoderBasic, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        # self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.rnn = BiAspectGRULayer(input_sz=embed_size, hidden_sz=hidden_size, asp_size=asp_size)

    def forward(self, src, src_lens, asp_hidden, src_mask=None, title=None, title_lens=None, title_mask=None):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        Other parameters will not be used in the RNNENcoderBasic class, they are here because we want to have a unify interface
        :return:
        """
        # Debug
        # if math.isnan(self.rnn.weight_hh_l0[0,0].item()):
        #    logging.info('nan encoder parameter')
        batch_sz = src.shape[0]
        src_embed = self.embedding(src)  # [batch, src_len, embed_size]

        # src_lens = torch.as_tensor(src_lens, dtype=torch.int64, device=src_embed.device)
        # lengths, sorted_indices = torch.sort(src_lens, descending=True)
        # unsorted_indices = invert_permutation(sorted_indices)

        state = torch.zeros(batch_sz, self.hidden_size, dtype=src_embed.dtype, device=src_embed.device)

        # src_embed = src_embed.index_select(0, sorted_indices)
        memory_bank, encoder_final_state = self.rnn(src_embed, state, asp_hidden)
        # memory_bank = memory_bank.index_select(0, unsorted_indices)
        # encoder_final_state = encoder_final_state.index_select(1, unsorted_indices)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1, :, :], encoder_final_state[-2, :, :]),
                                                       1)  # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :]  # [batch, hidden_size]

        return memory_bank.contiguous(), encoder_last_layer_final_state
