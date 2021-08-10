import torch
import argparse
import config
import logging
import os
import json
from data_process.io import KeyphraseDataset
from torch.optim import Adam
from model import Seq2SeqModel
from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab
import time
import numpy as np
import random
import data_process
import train_ml


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    if hasattr(opt, 'train_ml') and opt.train_ml:
        opt.exp += '.ml'

    if hasattr(opt, 'train_rl') and opt.train_rl:
        opt.exp += '.rl'

    if opt.one2many:
        opt.exp += '.one2many'

    if opt.one2many_mode == 1:
        opt.exp += '.cat'

    if opt.copy_attention:
        opt.exp += '.copy'

    if opt.coverage_attn:
        opt.exp += '.coverage'

    if opt.review_attn:
        opt.exp += '.review'

    if opt.orthogonal_loss:
        opt.exp += '.orthogonal'

    if opt.use_target_encoder:
        opt.exp += '.target_encode'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    opt.exp += f'.{opt.asp_fusion_mode}'

    if opt.snippet_loss:
        opt.exp += '.snippet_loss'

    if opt.sentiment_loss:
        opt.exp += '.sentiment_loss'

    if opt.snippet_copy:
        opt.exp += '.snippet_copy'

    if opt.sentiment_gen:
        opt.exp += '.sentiment_gen'

    if opt.include_peos:
        opt.exp += '.include_peos'

    if opt.delimiter_type == 0:
        opt.delimiter_word = data_process.io.SEP_WORD
    else:
        opt.delimiter_word = data_process.io.EOS_WORD

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    logging.info('EXP_PATH : ' + opt.exp_path)

    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'wb')
                   )
        json.dump(vars(opt), open(os.path.join(opt.model_path, opt.exp + '.initial.json'), 'w'))

    return opt


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')

    model = Seq2SeqModel(opt)

    if opt.pretrained_model != "":
        logging.info(f'load pretrained model from {opt.pretrained_model}')
        save_model = torch.load(opt.pretrained_model)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys() and model_dict[k].shape == v.shape}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    return model.to(opt.device)


def init_optimizer_criterion(model, opt):
    """
    mask the PAD <pad> when computing loss, before we used weight matrix, but not handy for copy-model, change to ignore_index
    :param model:
    :param opt:
    :return:
    """

    optimizer_ml = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)

    return optimizer_ml


def main(opt):
    start_time = time.time()
    train_data_loader, valid_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=True)

    load_data_time = time_since(start_time)
    logging.info('Time for loading the data: %.1f' % load_data_time)
    start_time = time.time()
    model = init_model(opt)
    optimizer_ml = init_optimizer_criterion(model, opt)
    train_ml.train_model(model, optimizer_ml, train_data_loader, valid_data_loader, opt)
    # train_ml.train_model(model, optimizer_ml, train_data_loader, valid_data_loader, opt)
    training_time = time_since(start_time)
    logging.info('Time for training: %.1f' % training_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()

    opt.exp_path = 'exp/%s.%s'
    opt.exp = 'abkg'
    opt.epochs = 200
    opt.copy_attention = True
    opt.batch_size = 256
    opt.seed = 2333

    opt.tag_embed_size = 30
    opt.asp_embed_size = 50
    opt.sentiment_embed_size = opt.word_vec_size
    opt.sentiment_hidden_size = opt.word_vec_size

    opt.early_stop_tolerance = 10
    opt.checkpoint_interval = 50

    opt.snippet_weight = 1000
    opt.trg_weight = 1
    opt.sentiment_weight = 20

    opt.peos_idx = 4

    opt.asp_fusion_mode = 'attn_multi'
    # opt.sentiment_gen = True
    opt.include_peos = True

    opt = process_opt(opt)

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
