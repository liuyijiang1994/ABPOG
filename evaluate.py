# from nltk.stem.porter import *
import torch
# from utils import Progbar
# from pykp.metric.bleu import bleu
from masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics, RewardStatistics
import time
from utils.time_log import time_since
# from nltk.stem.porter import *
import logging
import numpy as np
from collections import defaultdict
import os
import sys
from utils.string_helper import *
import data_process
import json
from sklearn.metrics import classification_report
import math


# stemmer = PorterStemmer()

def evaluate_snippet_loss(data_loader, model, opt):
    model.eval()
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0
    snippet_loss_sum = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            src, src_len, src_mask, trg, trg_len, trg_mask, snippet, snippet_len, snippet_mask, aspect_id, sentiment_id, _ = batch

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            snippet = snippet.to(opt.device)
            snippet_mask = snippet_mask.to(opt.device)
            aspect_id = aspect_id.to(opt.device)

            start_time = time.time()

            snippet_logit = model.snippet_forward(src, src_len, src_mask, aspect_id)

            forward_time = time_since(start_time)
            forward_time_total += forward_time
            start_time = time.time()
            snippet_loss = model.snippet_loss(snippet_logit, snippet, snippet_mask)
            if math.isnan(snippet_loss.item()):
                print('nan1', batch_i, n_batch)

            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time
            snippet_loss_sum += snippet_loss.item()

    if math.isnan(snippet_loss_sum / n_batch):
        print('nan2', batch_i, n_batch)
    return snippet_loss_sum / n_batch


def evaluate_loss(data_loader, model, opt):
    model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            src, src_len, src_mask, trg, trg_len, trg_mask, snippet, snippet_len, snippet_mask, aspect_id, sentiment_id, _ = batch

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            aspect_id = aspect_id.to(opt.device)

            start_time = time.time()
            decoder_dist, h_t, attention_dist, encoder_final_state, coverage = model(
                src, src_len, src_mask, trg, trg_len, trg_mask, aspect_id)

            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()
            trg_loss = model.loss_func(decoder_dist,
                                       trg, trg_len, trg_mask)

            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += trg_loss.item()
            total_trg_tokens += sum(trg_len)

    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total,
                                    loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def preprocess_beam_search_result(beam_search_result, idx2word, eos_idx):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best in zip(predictions, scores, attention):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, eos_idx)
            # sentence = [idx2word[int(idx.item())] if int(idx.item()) < vocab_size else oov[int(idx.item())-vocab_size] for idx in pred[:-1]]
            sentences_n_best.append(sentence)
        pred_dict[
            'sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
        pred_dict[
            'attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list


def get_snippet(snippet):
    # 00001001111110000
    snippet_list = []
    start = -1
    for idx, tag in enumerate(snippet):
        if tag == 0:
            if start >= 0:
                snippet_list.append((start, idx))
                start = -1
        elif tag == 1:
            if start < 0:
                start = idx
    if start > 0:
        snippet_list.append((start, len(snippet) + 1))
    return snippet_list


def evaluate_snippet(model, one2many_data_loader, opt):
    interval = 1000
    all_snippet = []
    all_pred_snippet = []
    with torch.no_grad():
        start_time = time.time()
        for batch_i, batch in enumerate(one2many_data_loader):
            if (batch_i + 1) % interval == 0:
                print("Batch %d: Time for running beam search on %d batches : %.1f" % (
                    batch_i + 1, interval, time_since(start_time)))
                sys.stdout.flush()
                start_time = time.time()
            src, src_lens, src_mask, trg, trg_len, trg_mask, snippet, snippet_len, snippet_mask, aspect_id, sentiment_id, original_idx_list = batch
            """
            src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
            src_lens: a list containing the length of src sequences for each batch, with len=batch
            src_mask: a FloatTensor, [batch, src_seq_len]
            """
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            aspect_id = aspect_id.to(opt.device)
            snippet_logit = model.snippet_forward(src, src_lens, src_mask, aspect_id)
            pred_snippet = torch.argmax(snippet_logit, dim=-1)

            src = src.cpu().numpy().tolist()
            snippet = snippet.cpu().numpy().tolist()
            pred_snippet = pred_snippet.cpu().numpy().tolist()
            aspect_id = aspect_id.cpu().numpy().tolist()
            # recover the original order in the dataset
            seq_pairs = sorted(zip(original_idx_list, src, src_lens,
                                   pred_snippet, snippet, aspect_id),
                               key=lambda p: p[0])
            original_idx_list, src, src_lens, pred_snippet, snippet, aspect_id = zip(*seq_pairs)
            # Process every src in the batch
            for s, slen, psni, sni, asp in zip(src, src_lens, pred_snippet, snippet, aspect_id):
                s = s[:slen]
                psni = psni[:slen]
                sni = sni[:slen]
                all_snippet.extend(sni)
                all_pred_snippet.extend(psni)
                s_word = [opt.idx2word[t] for t in s]
                t_pred_snip = [s_word[start:end] for start, end in get_snippet(psni)]
                t_true_snip = [s_word[start:end] for start, end in get_snippet(sni)]
    print(classification_report(all_snippet, all_pred_snippet))


def evaluate_beam_search(generator, one2many_data_loader, opt, delimiter_word='<sep>'):
    # score_dict_all = defaultdict(list)  # {'precision@5':[],'recall@5':[],'f1_score@5':[],'num_matches@5':[],'precision@10':[],'recall@10':[],'f1score@10':[],'num_matches@10':[]}
    # file for storing the predicted keyphrases
    pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
    # debug
    interval = 1000
    with torch.no_grad():
        start_time = time.time()
        for batch_i, batch in enumerate(one2many_data_loader):
            if (batch_i + 1) % interval == 0:
                print("Batch %d: Time for running beam search on %d batches : %.1f" % (
                    batch_i + 1, interval, time_since(start_time)))
                sys.stdout.flush()
                start_time = time.time()
            src_list, src_lens, src_mask, trg, trg_len, trg_mask, snippet, snippet_len, snippet_mask, aspect_id, sentiment_id, original_idx_list = batch
            """
            src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
            src_lens: a list containing the length of src sequences for each batch, with len=batch
            src_mask: a FloatTensor, [batch, src_seq_len]
            """
            src_list = src_list.to(opt.device)
            src_mask = src_mask.to(opt.device)
            aspect_id = aspect_id.to(opt.device)

            beam_search_result = generator.beam_search(src_list, src_lens, src_mask, opt.word2idx, aspect_id,
                                                       opt.max_eos_per_output_seq)
            pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word,
                                                      opt.word2idx[data_process.io.EOS_WORD])
            # list of {"sentences": [], "scores": [], "attention": []}

            # recover the original order in the dataset
            src_list = src_list.cpu().numpy().tolist()
            seq_pairs = sorted(zip(original_idx_list, src_list, src_lens, pred_list, aspect_id), key=lambda p: p[0])
            original_idx_list, src_list, src_lens, pred_list, aspect_id = zip(*seq_pairs)
            # Process every src in the batch
            for src, src_len, pred, aspid in zip(src_list, src_lens, pred_list, aspect_id):
                # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                # pred_seq_list: a list of sequence objects, sorted by scores
                # oov: a list of oov words

                # predicted sentences from a single src, a list of list of word, with len=[beam_size, out_seq_len], does not include the final <EOS>
                pred_str_list = pred['sentences']
                pred_score_list = pred['scores']
                # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
                pred_attn_list = pred['attention']

                # output the predicted keyphrases to a file
                pred_print_out = ''
                for word_list_i, word_list in enumerate(pred_str_list):
                    if word_list_i < len(pred_str_list) - 1:
                        pred_print_out += '%s;' % ''.join(word_list)
                    else:
                        pred_print_out += '%s' % ''.join(word_list)
                data = {'asp': aspid.item(), 'keyphrase': pred_print_out}
                data = json.dumps(data, ensure_ascii=False)
                pred_output_file.write(str(data) + '\n')
    pred_output_file.close()
    print("done!")
