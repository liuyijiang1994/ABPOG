import logging
import torch
import torch.nn as nn
import numpy as np
import random
from data_process import io
from asprnn_encoder import *
from hi_rnn_decoder import HiRNNDecoder
from torchcrf import CRF
import common
from attention import Attention
from masked_loss import masked_cross_entropy
import math
from masked_softmax import masked_softmax


class Seq2SeqModel(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqModel, self).__init__()

        self.vocab_size = opt.vocab_size
        self.emb_dim = opt.word_vec_size
        self.num_directions = 2 if opt.bidirectional else 1
        self.encoder_size = opt.encoder_size
        self.decoder_size = opt.decoder_size
        # self.ctx_hidden_dim = opt.rnn_size
        self.batch_size = opt.batch_size
        self.bidirectional = opt.bidirectional
        self.enc_layers = opt.enc_layers
        self.dec_layers = opt.dec_layers
        self.dropout = opt.dropout

        self.bridge = opt.bridge
        self.asp_fusion_mode = opt.asp_fusion_mode

        self.coverage_attn = opt.coverage_attn
        self.copy_attn = opt.copy_attention

        self.pad_idx_src = opt.word2idx[io.PAD_WORD]
        self.pad_idx_trg = opt.word2idx[io.PAD_WORD]
        self.bos_idx = opt.word2idx[io.BOS_WORD]
        self.eos_idx = opt.word2idx[io.EOS_WORD]
        self.unk_idx = opt.word2idx[io.UNK_WORD]

        self.share_embeddings = opt.share_embeddings
        self.review_attn = opt.review_attn

        self.attn_mode = opt.attn_mode
        self.snippet_copy = opt.snippet_copy

        self.device = opt.device

        self.snippet_weight = opt.snippet_weight
        self.trg_weight = opt.trg_weight

        self.encoder = RNNEncoderBasic(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            asp_size=opt.asp_embed_size,
            hidden_size=self.encoder_size,
            num_layers=self.enc_layers,
            bidirectional=self.bidirectional,
            pad_token=self.pad_idx_src,
            dropout=self.dropout
        )

        self.decoder = HiRNNDecoder(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.decoder_size,
            num_layers=self.dec_layers,
            memory_bank_size=self.num_directions * self.encoder_size,
            coverage_attn=self.coverage_attn,
            copy_attn=self.copy_attn,
            pad_idx=self.pad_idx_trg,
            attn_mode=self.attn_mode,
            asp_fusion_mode=opt.asp_fusion_mode,
            asp_num=len(common.asp_dict),
            asp_hidden_size=opt.asp_embed_size,
            snippet_copy=self.snippet_copy,
            dropout=self.dropout,
            sentiment_gen=opt.sentiment_gen,
            peos_idx=opt.peos_idx
        )
        self.sigmoid = nn.Sigmoid()

        if self.bridge == 'dense':
            self.bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.decoder_size)
        elif opt.bridge == 'dense_nonlinear':
            self.bridge_layer = nn.tanh(
                nn.Linear(self.encoder_size * self.num_directions, self.decoder_size))
        else:
            self.bridge_layer = None

        if self.bridge == 'copy':
            assert self.encoder_size * self.num_directions == self.decoder_size, 'encoder hidden size and decoder hidden size are not match, please use a bridge layer'

        if self.share_embeddings:
            self.encoder.embedding.weight = self.decoder.embedding.weight

        self.asp_embedding = nn.Embedding(len(common.asp_dict), opt.asp_embed_size)
        self.asp_fusion_mode = opt.asp_fusion_mode

        self.asp_attn = Attention(opt.asp_embed_size, self.num_directions * self.encoder_size,
                                  coverage_attn=False, attn_mode='general')
        self.asp_sentiment_attn = Attention(opt.asp_embed_size, self.num_directions * self.encoder_size,
                                            coverage_attn=False, attn_mode='general')

        self.h_init_linear = nn.Linear(opt.asp_embed_size + self.num_directions * self.encoder_size, 1)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
        if not self.share_embeddings:
            self.decoder.embedding.weight.data.uniform_(-initrange, initrange)

        # TODO: model parameter init
        # fill with fixed numbers for debugging
        # self.embedding.weight.data.fill_(0.01)
        # self.encoder2decoder_hidden.bias.data.fill_(0)
        # self.encoder2decoder_cell.bias.data.fill_(0)
        # self.decoder2vocab.bias.data.fill_(0)

    def _encode(self, src, src_lens, src_mask, asp):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :return:
        """
        asp_hidden = self.asp_embedding(asp)

        batch_size, max_src_len = list(src.size())
        src_mask = src_mask.bool()
        # Encoding
        memory_bank, encoder_final_state = self.encoder(src, src_lens, asp_hidden, src_mask)
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        final_memory_bank = memory_bank

        return final_memory_bank, asp_hidden

    def _decode(self, memory_bank, encoder_final_state, src, src_mask, trg,
                aspect_id, asp_hiden):
        batch_size, max_src_len = list(src.size())

        h_t_init = self.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]
        max_target_length = trg.size(1)

        decoder_dist_all = []
        attention_dist_all = []

        if self.coverage_attn:
            coverage = torch.zeros_like(src, dtype=torch.float).requires_grad_()  # [batch, max_src_seq]
            coverage_all = []
        else:
            coverage = None
            coverage_all = None

        # init y_t to be BOS token
        y_t_init = trg.new_ones(batch_size) * self.bos_idx  # [batch_size]
        sentiment_state = torch.zeros(batch_size).to(memory_bank.device)
        weighted_asp_hidden = torch.zeros(batch_size, memory_bank.size()[-1]).to(memory_bank.device)
        for t in range(max_target_length):
            if t == 0:
                h_t = h_t_init
                y_t = y_t_init
            else:
                h_t = h_t_next
                y_t = y_t_next
            decoder_dist, final_dist, h_t_next, context, \
            attn_dist, p_gen, coverage, sentiment_state, weighted_asp_hidden = \
                self.decoder(y_t, h_t, memory_bank, src_mask,
                             src, aspect_id, asp_hiden, h_t_init, sentiment_state, coverage, weighted_asp_hidden)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]
            if self.coverage_attn:
                coverage_all.append(coverage.unsqueeze(1))  # [batch, 1, src_seq_len]
            y_t_next = trg[:, t]  # [batch]

        decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
        attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]
        if self.coverage_attn:
            coverage_all = torch.cat(coverage_all, dim=1)  # [batch_size, trg_len, src_len]
            assert coverage_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
        assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        return decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state, coverage_all

    def loss_func(self, decoder_dist_all, trg, trg_lens, trg_mask):
        trg_loss = masked_cross_entropy(decoder_dist_all, trg, trg_mask, trg_lens)

        return trg_loss

    def get_snippet_and_sentiment(self, attn_memory_bank, src_mask, asp_hidden):
        encoder_final_state = pool(attn_memory_bank, src_mask.unsqueeze(2), pool_type='avg')

        return encoder_final_state
        # return fused_encoder_final_state, sentiment_dist, transed_snippet

    def forward(self, src, src_len, src_mask, trg, trg_len, trg_mask, aspect):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :return:
        """
        attn_memory_bank, asp_hidden = self._encode(src, src_len, src_mask, aspect)

        encoder_final_state = self.get_snippet_and_sentiment(attn_memory_bank,
                                                             src_mask, asp_hidden)

        decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state, coverage_all = self._decode(
            attn_memory_bank, encoder_final_state, src, src_mask,
            trg, aspect, asp_hidden)

        return decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state, coverage_all

    def predict(self, src, src_len, src_mask, aspect):
        attn_memory_bank, asp_hidden = self._encode(src, src_len, src_mask, aspect)

        encoder_final_state = self.get_snippet_and_sentiment(
            attn_memory_bank,
            src_mask, asp_hidden)

        return attn_memory_bank, encoder_final_state, asp_hidden

    def init_decoder_state(self, encoder_final_state):
        """
        :param encoder_final_state: [batch_size, self.num_directions * self.encoder_size]
        :return: [1, batch_size, decoder_size]
        """
        batch_size = encoder_final_state.size(0)
        if self.bridge == 'none':
            decoder_init_state = None
        elif self.bridge == 'copy':
            decoder_init_state = encoder_final_state
        else:
            decoder_init_state = self.bridge_layer(encoder_final_state)
        decoder_init_state = decoder_init_state.expand((batch_size, self.decoder_size))
        # [dec_layers, batch_size, decoder_size]
        return decoder_init_state


def pad_sequence(xs, length):
    tag_ids = np.full((len(xs), length), 0)
    for idx, x in enumerate(xs):
        tag_ids[idx][:len(x)] = x

    return torch.from_numpy(tag_ids)


def pool(h, mask, pool_type='max'):
    mask = 1 - mask
    mask = mask.bool()
    if pool_type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, 1)[0]
    elif pool_type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)
