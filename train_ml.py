import torch.nn as nn
from masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
from utils.time_log import time_since
from evaluate import evaluate_loss
import time
import math
import logging
import torch
import sys
import os
from utils.report import export_train_and_valid_loss

EPS = 1e-8


def train_model(model, optimizer_ml, train_data_loader, valid_data_loader, opt):
    logging.info('======================  Start Training  =========================')

    total_batch = -1
    early_stop_flag = False

    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss = float('inf')
    num_stop_dropping = 0
    sample_num = 0
    model.train()

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        if early_stop_flag:
            break

        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1
            sample_num += len(batch[0])

            # Training
            batch_loss_stat, decoder_dist = train_one_batch(batch, model, optimizer_ml,
                                                            opt, batch_i)

            report_train_loss_statistics.update(batch_loss_stat)
            total_train_loss_statistics.update(batch_loss_stat)

            # Brief report
            '''
            if batch_i % opt.report_every == 0:
                brief_report(epoch, batch_i, one2one_batch, loss_ml, decoder_log_probs, opt)
            '''

            # Checkpoint, decay the learning rate if validation loss stop dropping, apply early stopping if stop decreasing for several epochs.
            # Save the model parameters if the validation loss improved.
            if total_batch % 4000 == 0:
                print()
                print("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
                sys.stdout.flush()

            if total_batch % opt.checkpoint_interval == 0:

                # test the model on the validation dataset for one epoch
                valid_loss_stat = evaluate_loss(valid_data_loader, model, opt)
                model.train()
                current_valid_loss = valid_loss_stat.xent()
                current_valid_ppl = valid_loss_stat.ppl()
                print("Enter check point!")
                sys.stdout.flush()

                current_train_ppl = report_train_loss_statistics.ppl()
                current_train_loss = report_train_loss_statistics.xent()

                # debug
                if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
                    logging.info(
                        "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (
                            epoch, batch_i, total_batch))
                    exit()

                if current_valid_loss < best_valid_loss:  # update the best valid loss and save the model parameters
                    print("Valid loss drops")
                    sys.stdout.flush()
                    best_valid_loss = current_valid_loss
                    best_valid_ppl = current_valid_ppl
                    num_stop_dropping = 0

                    # check_pt_model_path = os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (
                    #     opt.exp, epoch, batch_i, total_batch) + '.model')
                    check_pt_model_path = os.path.join(opt.model_path, '%s' % (opt.exp) + '.model')
                    torch.save(  # save model parameters
                        model.state_dict(), open(check_pt_model_path, 'wb')
                    )
                    logging.info('Saving checkpoint to %s' % check_pt_model_path)
                else:
                    print("Valid loss does not drop")
                    sys.stdout.flush()
                    num_stop_dropping += 1
                    # decay the learning rate by a factor
                    for i, param_group in enumerate(optimizer_ml.param_groups):
                        old_lr = float(param_group['lr'])
                        new_lr = old_lr * opt.learning_rate_decay
                        if old_lr - new_lr > EPS:
                            param_group['lr'] = new_lr

                # log loss, ppl, and time
                # print("check point!")
                # sys.stdout.flush()
                logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                logging.info(
                    'avg training ppl: %.3f; avg validation ppl: %.3f; best validation ppl: %.3f' % (
                        current_train_ppl, current_valid_ppl, best_valid_ppl))
                logging.info(
                    'avg training trg loss: %.3f; avg validation trg loss: %.3f; best validation trg loss: %.3f' % (
                        current_train_loss, current_valid_loss, best_valid_loss))

                report_train_ppl.append(current_train_ppl)
                report_valid_ppl.append(current_valid_ppl)
                report_train_loss.append(current_train_loss)
                report_valid_loss.append(current_valid_loss)

                if num_stop_dropping >= opt.early_stop_tolerance:
                    logging.info(
                        'Have not increased for %d check points, early stop training' % num_stop_dropping)
                    early_stop_flag = True
                    break
                logging.info('\n')
                report_train_loss_statistics.clear()
                sample_num = 0

    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_loss(report_train_loss, report_valid_loss, report_train_ppl, report_valid_ppl,
                                opt.checkpoint_interval, train_valid_curve_path)
    # logging.info('Overall average training loss: %.3f, ppl: %.3f' % (total_train_loss_statistics.xent(), total_train_loss_statistics.ppl()))


def train_one_batch(batch, model, optimizer, opt, batch_i):
    with torch.autograd.set_detect_anomaly(True):
        src, src_len, src_mask, trg, trg_len, trg_mask, snippet, snippet_len, snippet_mask, aspect_id, sentiment_id, _ = batch
        """
        src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        src_lens: a list containing the length of src sequences for each batch, with len=batch
        src_mask: a FloatTensor, [batch, src_seq_len]
        trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        trg_lens: a list containing the length of trg sequences for each batch, with len=batch
        trg_mask: a FloatTensor, [batch, trg_seq_len]
        """

        # move data to GPU if available
        src = src.to(opt.device)
        src_mask = src_mask.to(opt.device)
        trg = trg.to(opt.device)
        trg_mask = trg_mask.to(opt.device)
        aspect_id = aspect_id.to(opt.device)

        optimizer.zero_grad()

        start_time = time.time()

        decoder_dist, h_t, attention_dist, encoder_final_state, coverage = model(
            src, src_len, src_mask, trg, trg_len, trg_mask, aspect_id)

        forward_time = time_since(start_time)

        start_time = time.time()
        trg_loss = model.loss_func(decoder_dist, trg, trg_len, trg_mask)

        loss_compute_time = time_since(start_time)

        total_trg_tokens = sum(trg_len)

        if math.isnan(trg_loss.item()):
            print("Batch i: %d" % batch_i)
            print("src")
            print(src)
            print(src_len)
            print(src_mask)
            print("trg")
            print(trg)
            print(trg_len)
            print(trg_mask)
            print("Decoder")
            print(decoder_dist)
            print(h_t)
            print(attention_dist)
            raise ValueError("Loss is NaN")

        if opt.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
            normalization = total_trg_tokens
        elif opt.loss_normalization == 'batches':  # use batch_size to normalize the loss
            normalization = src.size(0)
        else:
            raise ValueError('The type of loss normalization is invalid.')

        assert normalization > 0, 'normalization should be a positive number'

        start_time = time.time()
        # back propagation on the normalized loss
        loss = trg_loss.div(normalization)

        loss.backward()
        backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        # grad_norm_after_clipping = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
        # logging.info('clip grad (%f -> %f)' % (grad_norm_before_clipping, grad_norm_after_clipping))

    optimizer.step()

    # construct a statistic object for the loss
    stat = LossStatistics(trg_loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time,
                          loss_compute_time=loss_compute_time, backward_time=backward_time)

    return stat, decoder_dist.detach()
