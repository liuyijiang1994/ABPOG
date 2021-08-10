# -*- coding: utf-8 -*-
"""
Python File Template
Built on the source code of seq2seq-keyphrase-pytorch: https://github.com/memray/seq2seq-keyphrase-pytorch
"""
import numpy as np
import torch
import torch.utils.data

import common

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
SEP_WORD = '<sep>'
DIGIT = '<digit>'
PEOS_WORD = '<peos>'


class KeyphraseDataset(torch.utils.data.Dataset):
    def __init__(self, examples, word2idx, idx2word, load_train, include_peos=False):
        # keys of matter. `src_oov_map` is for mapping pointed word to dict, `oov_dict` is for determining the dim of predicted logit: dim=vocab_size+max_oov_dict_in_batch

        self.examples = examples
        self.word2idx = word2idx
        self.id2xword = idx2word
        self.pad_idx = word2idx[PAD_WORD]
        self.include_peos = include_peos
        self.load_train = load_train

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad(self, input_list, pad_idx):
        input_list_lens = [len(l) for l in input_list]
        max_seq_len = max(input_list_lens)
        padded_batch = pad_idx * np.ones((len(input_list), max_seq_len))

        for j in range(len(input_list)):
            current_len = input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j]

        padded_batch = torch.LongTensor(padded_batch)

        input_mask = torch.ne(padded_batch, pad_idx)
        input_mask = input_mask.type(torch.ByteTensor)

        return padded_batch, input_list_lens, input_mask

    def collate_fn_one2one(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        # source with oov words replaced by <unk>
        src_word_id_list = [b['src_word_id_list'] for b in batches]
        trg_word_id_list = [b['keyphrase_word_id_list'] + [self.word2idx[EOS_WORD]] for b in batches]

        snippet_tag_id_list = [b['snippet_tag_id'] for b in batches]
        asp_id = [b['asp_id'] for b in batches]
        sentiment_id = [b['sentiment'] for b in batches]

        original_indices = list(range(len(batches)))
        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        seq_pairs = sorted(
            zip(src_word_id_list, trg_word_id_list, snippet_tag_id_list, asp_id, sentiment_id, original_indices),
            key=lambda p: len(p[0]), reverse=True)
        src_word_id_list, trg_word_id_list, snippet_tag_id_list, asp_id, sentiment_id, original_indices = zip(
            *seq_pairs)

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src_id, src_lens, src_mask = self._pad(src_word_id_list, self.pad_idx)
        trg_id, trg_lens, trg_mask = self._pad(trg_word_id_list, self.pad_idx)
        tag_id, tag_lens, tag_mask = self._pad(snippet_tag_id_list, 2)
        asp_id = torch.LongTensor(asp_id)
        sentiment_id = torch.LongTensor(sentiment_id)

        return src_id, src_lens, src_mask, trg_id, trg_lens, trg_mask, tag_id, tag_lens, tag_mask, asp_id, sentiment_id, original_indices


def build_dataset(src_trgs_pairs, word2idx, opt):
    '''
    Standard process for copy model
    :param include_original: keep the original texts of source and target
    :return:
    '''
    for idx, data in enumerate(src_trgs_pairs):
        # if w is not seen in training data vocab (word2idx, size could be larger than opt.vocab_size), replace with <unk>
        # if w's id is larger than opt.vocab_size, replace with <unk>

        data['src_word_id_list'] = [
            word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in
            data['src_word_list']]
        if not opt.include_peos:
            data['keyphrase_word_id_list'] = [
                word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in
                data['keyphrase_word_list']]
        else:
            data['keyphrase_word_id_list'] = [
                word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in
                data['peos_keyphrase_word_list']]
        if idx % 20000 == 0:
            for k, v in data.items():
                print(k, v)
            print('-' * 20)
            print()

    return src_trgs_pairs
