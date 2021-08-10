import torch
import logging
from data_process.io import KeyphraseDataset
from torch.utils.data import DataLoader


def load_vocab(opt):
    # load vocab
    logging.info("Loading vocab from disk: %s" % (opt.vocab))
    word2idx, idx2word, vocab = torch.load(opt.vocab + '/vocab.pt', 'wb')
    # assign vocab to opt
    opt.word2idx = word2idx
    opt.idx2word = idx2word
    opt.vocab = vocab
    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return word2idx, idx2word, vocab


def load_data_and_vocab(opt, load_train=True):
    # load vocab
    word2idx, idx2word, vocab = load_vocab(opt)

    # constructor data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)

    if load_train:  # load training dataset
        train_data = torch.load(opt.data + '/train_dataset.pt', 'wb')
        train_dataset = KeyphraseDataset(train_data, word2idx=word2idx, idx2word=idx2word,
                                         include_peos=opt.include_peos, load_train=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  collate_fn=train_dataset.collate_fn_one2one,
                                  num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                  shuffle=True)
        logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

        valid_data = torch.load(opt.data + '/valid_dataset.pt', 'wb')

        valid_dataset = KeyphraseDataset(valid_data, word2idx=word2idx, idx2word=idx2word,
                                         include_peos=opt.include_peos, load_train=True)
        valid_loader = DataLoader(dataset=valid_dataset,
                                  collate_fn=valid_dataset.collate_fn_one2one,
                                  num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                  shuffle=False)

        logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))
        return train_loader, valid_loader, word2idx, idx2word, vocab

    else:
        test_data = torch.load(opt.data + '/test_dataset.pt', 'wb')
        test_dataset = KeyphraseDataset(test_data, word2idx=word2idx, idx2word=idx2word,
                                        include_peos=opt.include_peos, load_train=False)
        test_loader = DataLoader(dataset=test_dataset,
                                 collate_fn=test_dataset.collate_fn_one2one,
                                 num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                 shuffle=False)
        logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))

        return test_loader, word2idx, idx2word, vocab
