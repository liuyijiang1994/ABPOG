import logging
import torch
import torch.nn as nn
from attention import Attention
import numpy as np
from masked_softmax import masked_softmax
import math
import logging
from sentiGRU import SentimentGRUCell


class HiRNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, sentiment_gen, num_layers,
                 memory_bank_size, coverage_attn, copy_attn, pad_idx, attn_mode, asp_fusion_mode, asp_num,
                 asp_hidden_size, snippet_copy=False, dropout=0.0, peos_idx=4):
        super(HiRNNDecoder, self).__init__()
        # self.input_size = input_size
        # self.input_size = embed_size + memory_bank_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.memory_bank_size = memory_bank_size
        self.dropout = nn.Dropout(dropout)
        self.coverage_attn = coverage_attn
        self.copy_attn = copy_attn
        self.pad_token = pad_idx
        self.snippet_copy = snippet_copy
        self.asp_fusion_mode = asp_fusion_mode
        self.asp_num = asp_num
        self.sentiment_gen = sentiment_gen
        self.peos_idx = peos_idx
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        # if self.sentiment_gen:
        #     self.input_size = embed_size + sentiment_embed_size
        # else:
        self.input_size = embed_size
        if self.sentiment_gen:
            self.rnn = SentimentGRUCell(input_size=self.input_size, hidden_size=hidden_size,
                                        senti_size=memory_bank_size)
        else:
            self.rnn = nn.GRUCell(input_size=self.input_size, hidden_size=hidden_size)

        self.global_rnn = nn.GRUCell(input_size=self.hidden_size, hidden_size=hidden_size)

        self.attention_layer = Attention(
            decoder_size=hidden_size,
            memory_bank_size=memory_bank_size,
            coverage_attn=coverage_attn,
            attn_mode=attn_mode
        )

        self.asp_attention_layer = Attention(
            decoder_size=hidden_size + asp_hidden_size,
            memory_bank_size=memory_bank_size,
            coverage_attn=coverage_attn,
            attn_mode=attn_mode
        )

        if copy_attn:
            p_gen_input_size = embed_size + hidden_size + memory_bank_size
            self.p_gen_linear = nn.Linear(p_gen_input_size, 1)

        self.sigmoid = nn.Sigmoid()
        # self.p_gen_linear = nn.Linear(input_size + hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()
        # self.vocab_dist_network = nn.Sequential(nn.Linear(hidden_size + memory_bank_size, hidden_size), nn.Linear(hidden_size, vocab_size), nn.Softmax(dim=1))

        if self.asp_fusion_mode.endswith('single'):
            self.vocab_dist_linear_1 = nn.Linear(hidden_size + memory_bank_size, hidden_size)
            self.vocab_dist_linear_2 = nn.Linear(hidden_size, vocab_size)
        elif self.asp_fusion_mode.endswith('multi'):
            self.vocab_dist_linear_1 = nn.ModuleList(
                [nn.Linear(hidden_size + memory_bank_size, hidden_size) for i in range(asp_num)])
            self.vocab_dist_linear_2 = nn.ModuleList(
                [nn.Linear(hidden_size, vocab_size) for i in range(asp_num)])
        self.p_vocab_snippet = nn.Linear(hidden_size + memory_bank_size + asp_hidden_size, 1)

    def forward(self, y, h, memory_bank, src_mask,
                src, aspect_id, asp_hidden, h_t_init, sentiment_state, coverage, weighted_asp_hidden):
        """
        :param y: [batch_size]
        :param h: [num_layers, batch_size, decoder_size]
        :param memory_bank: [batch_size, max_src_seq_len, memory_bank_size]
        :param weighted_asp_hidden: [batch_size, memory_bank_size]
        :param src_mask: [batch_size, max_src_seq_len]
        :param coverage: [batch_size, max_src_seq_len]
        :param sentiment_state: [batch_size]
        :return:
        """
        batch_size, max_src_seq_len = list(src.size())
        assert y.size() == torch.Size([batch_size])
        assert h.size() == torch.Size([batch_size, self.hidden_size])

        # init input embedding
        y_emb = self.embedding(y)  # [batch_size, embed_size]
        # pass the concatenation of the input embedding and context vector to the RNN
        # insert one dimension to the context tensor
        # rnn_input = torch.cat((y_emb, context.unsqueeze(0)), 2)  # [1, batch_size, embed_size + num_directions * encoder_size]

        idx = []
        for i, v in enumerate(y.split(1, dim=0)):
            if v == self.peos_idx:
                idx.append(i)
                sentiment_state[i] = 1

        # 有输入为EOS，此时更新一下ht再放回去
        if len(idx) > 0:
            idx_tensor = torch.LongTensor(idx).to(y.get_device())
            mask = torch.zeros(batch_size).to(y.get_device())
            mask = mask.scatter(0, idx_tensor, torch.ones(len(idx)).to(y.get_device()))
            mask = mask.unsqueeze(1).repeat(1, self.hidden_size).bool()
            t_h_next = self.global_rnn(h, h_t_init)
            h = h.masked_scatter(mask, t_h_next)

            if self.sentiment_gen:
                target_attn_input = torch.cat((h, asp_hidden), dim=-1)
                t_weighted_asp_hidden, _, _ = self.asp_attention_layer(target_attn_input, memory_bank,
                                                                       src_mask=src_mask)
                weighted_asp_hidden = weighted_asp_hidden.masked_scatter(mask, t_weighted_asp_hidden)

        rnn_input = y_emb
        if self.sentiment_gen:
            # for i in [rnn_input, h, sentiment_embed, sentiment_hidden, sentiment_state]:
            #     print(i.shape)
            h_next = self.rnn(rnn_input, h, weighted_asp_hidden, sentiment_state)
        else:
            h_next = self.rnn(rnn_input, h)

        assert h_next.size() == torch.Size([batch_size, self.hidden_size])
        last_layer_h_next = h_next

        # apply attention, get input-aware context vector, attention distribution and update the coverage vector
        context, attn_dist, coverage = self.attention_layer(last_layer_h_next, memory_bank, src_mask, coverage)
        # context: [batch_size, memory_bank_size], attn_dist: [batch_size, max_input_seq_len], coverage: [batch_size, max_input_seq_len]
        assert context.size() == torch.Size([batch_size, self.memory_bank_size])
        assert attn_dist.size() == torch.Size([batch_size, max_src_seq_len])
        if self.coverage_attn:
            assert coverage.size() == torch.Size([batch_size, max_src_seq_len])

        vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)  # [B, memory_bank_size + decoder_size]

        if self.asp_fusion_mode.endswith('single'):
            vocab_dist = masked_softmax(
                self.vocab_dist_linear_2(self.dropout(self.vocab_dist_linear_1(vocab_dist_input))))
        elif self.asp_fusion_mode.endswith('multi'):
            vocab_dist = []
            for b_vocab_dist_input, b_aspect_id in zip(vocab_dist_input, aspect_id):
                b_aspect_id = b_aspect_id.item()
                t = self.vocab_dist_linear_2[b_aspect_id](
                    self.dropout(self.vocab_dist_linear_1[b_aspect_id](b_vocab_dist_input)))
                vocab_dist.append(t)
            vocab_dist = torch.stack(vocab_dist, dim=0)
            vocab_dist = masked_softmax(vocab_dist)

        p_gen = None
        if self.copy_attn:
            # [B, memory_bank_size + decoder_size + embed_size]
            p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)
            # p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))
            p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))

            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            final_dist = vocab_dist_.scatter_add(1, src, attn_dist_)
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size])

        return vocab_dist, final_dist, h_next, context, attn_dist, p_gen, coverage, sentiment_state, weighted_asp_hidden
