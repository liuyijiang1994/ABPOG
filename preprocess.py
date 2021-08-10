import argparse
from collections import Counter
import torch
import pickle
import data_process.io
import config
import json
from collections import defaultdict
import common


def get_tag(word_list, snippet_list):
    idx = word_list.find(snippet_list)
    tag = ['O'] * len(word_list)
    tag[idx] = 'B'
    for i in range(idx + 1, idx + len(snippet_list) - 1):
        tag[i] = 'I'
    tag[idx + len(snippet_list) - 1] = 'E'
    return tag


def read_src_and_trg_files(src_file):
    data_list = []
    for line_idx, line in enumerate(open(src_file, 'r')):
        data = json.loads(line)
        data['src_word_list'] = list(data['content'])
        data['keyphrase_word_list'] = list(data['keyphrase'])
        data['peos_keyphrase_word_list'] = list(data['target']) + ['<peos>'] + list(data['opinion'])
        data['snippet_word_list'] = list(data['snippet'])
        data['asp_id'] = common.asp_dict[data['asp']]

        assert len(data['src_word_list']) == len(list(data['content']))
        assert len(data['snippet_word_list']) == len(list(data['snippet_word_list']))

        data['snippet_tag'] = get_tag(data['content'], data['snippet'])
        data['snippet_tag_id'] = [0 if i == 'O' else 1 for i in data['snippet_tag']]
        data_list.append(data)

    return data_list


def build_vocab(tokenized_src_trg_pairs, include_peos):
    token_freq_counter = Counter()
    for data in tokenized_src_trg_pairs:
        token_freq_counter.update(data['src_word_list'])
        token_freq_counter.update(data['keyphrase_word_list'])

    # Discard special tokens if already present
    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
    if include_peos:
        special_tokens.append('<peos>')
    num_special_tokens = len(special_tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter:
            del token_freq_counter[s_t]

    word2idx = dict()
    idx2word = dict()
    for idx, word in enumerate(special_tokens):
        # '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3, '<peos>': 4
        word2idx[word] = idx
        idx2word[idx] = word

    sorted_word2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)

    sorted_words = [x[0] for x in sorted_word2idx]
    print('vocab_len', len(sorted_words))

    for idx, word in enumerate(sorted_words):
        word2idx[word] = idx + num_special_tokens

    for idx, word in enumerate(sorted_words):
        idx2word[idx + num_special_tokens] = word

    return word2idx, idx2word, token_freq_counter


def main(opt):
    # Preprocess training data
    """
    # Tokenize train_src and train_trg

    """

    # Tokenize train_src and train_trg, return a list of tuple, (src_word_list, [trg_1_word_list, trg_2_word_list, ...])
    tokenized_train_pairs = read_src_and_trg_files(opt.train_src)

    # build vocab from training src
    # build word2id, id2word, and vocab, where vocab is a counter
    # with special tokens, '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3
    # word2id, id2word are ordered by frequencies, includes all the tokens in the data
    # simply concatenate src and target when building vocab
    word2idx, idx2word, token_freq_counter = build_vocab(tokenized_train_pairs, opt.include_peos)

    train_dataset = data_process.io.build_dataset(tokenized_train_pairs, word2idx, opt)
    print("Dumping train_dataset to disk: %s" % (opt.data_dir + 'train_dataset.pt'))
    torch.save(train_dataset, open(opt.data_dir + 'train_dataset.pt', 'wb'))
    train_len = len(train_dataset)
    del train_dataset

    # Preprocess validation data
    tokenized_valid_pairs = read_src_and_trg_files(opt.valid_src)
    valid_dataset = data_process.io.build_dataset(tokenized_valid_pairs, word2idx, opt)
    print("Dumping valid to disk: %s" % (opt.data_dir + 'valid_dataset.pt'))
    torch.save(valid_dataset, open(opt.data_dir + 'valid_dataset.pt', 'wb'))
    valid_len = len(valid_dataset)
    del valid_dataset

    # Preprocess test data
    tokenized_test_pairs = read_src_and_trg_files(opt.test_src)
    test_dataset = data_process.io.build_dataset(tokenized_test_pairs, word2idx, opt)
    print("Dumping test to disk: %s" % (opt.data_dir + 'test_dataset.pt'))
    torch.save(test_dataset, open(opt.data_dir + 'test_dataset.pt', 'wb'))
    test_len = len(test_dataset)
    del test_dataset

    print("Dumping dict to disk: %s" % opt.data_dir + 'vocab.pt')
    torch.save([word2idx, idx2word, token_freq_counter],
               open(opt.data_dir + 'vocab.pt', 'wb'))
    print(f'Vocabulary size: {len(word2idx)}')
    print('#pairs of train_dataset  = %d' % train_len)
    print('#pairs of valid_dataset  = %d' % valid_len)
    print('#pairs of test_dataset = %d' % test_len)

    print('Done!')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # The source files are tokenized and the tokens are separated by a space character.
    # The target sequences in the target files are separated by ';' character

    parser.add_argument('-data_dir', help='The source file of the data')
    parser.add_argument('-include_peos', action="store_true", default=True, help='Include <peos> as a special token')

    config.vocab_opts(parser)
    opt = parser.parse_args()
    opt.include_peos = True
    # opt = vars(args) # convert to dict
    # data_dir = opt.data_dir + '/sample.'
    # data_dir = opt.data_dir + '/'
    data_dir = './data/'
    opt.data_dir = data_dir
    opt.train_src = data_dir + 'train.json'
    opt.valid_src = data_dir + 'valid.json'
    opt.test_src = data_dir + 'test.json'
    main(opt)
