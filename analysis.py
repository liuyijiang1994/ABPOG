import numpy as np
import common
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
from word_similarity import WordSimilarity2010

ws_tool = WordSimilarity2010()

kp_res = np.zeros((len(common.asp_dict), 2))
target_res = np.zeros((len(common.asp_dict), 2))
opinion_res = np.zeros((len(common.asp_dict), 2))
opinion_sim_res = np.zeros((len(common.asp_dict), 2))
kp_sim_res = np.zeros((len(common.asp_dict), 2))
triplet_sim_res = np.zeros((len(common.asp_dict), 2))
triplet_res = np.zeros((len(common.asp_dict), 2))

true_data = []
pred_data = []

true_sentiment = {idx: [] for idx, tag in common.idx2asp.items()}
pre_sentiment = {idx: [] for idx, tag in common.idx2asp.items()}
all_true_sentiment = []
all_pre_sentiment = []

bingo = 0
target_bingo = 0
opinion_bingo = 0
kp_sim_bingo = 0
opinion_sim_bingo = 0
triplet_bingo = 0
triplet_sim_bingo = 0

with open('data/test_kps.txt', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        true_data.append(data)

with open(
        # 'pred/predict.kp20k.copy.bi-directional.attn_multi.sentiment_loss.include_peos.20200721-095348/predictions.txt',
        'pred/predict.kp20k.copy.bi-directional.attn_multi.sentiment_loss.sentiment_gen.include_peos.20200720-083843/predictions.txt',
        'r') as f:
    for line in f:
        data = json.loads(line.strip())
        pred_data.append(data)

for td, pd in zip(true_data, pred_data):
    asp = common.idx2asp[td['asp']]
    true_sentiment[td['asp']].append(td['sentiment'])
    pre_sentiment[td['asp']].append(pd['sentiment'])
    all_true_sentiment.append(td['sentiment'])
    all_pre_sentiment.append(pd['sentiment'])

    if td['keyphrase'] == pd['keyphrase']:
        kp_res[td['asp']][0] += 1
        bingo += 1
    kp_res[td['asp']][1] += 1

    td_items = td['keyphrase'].split('<peos>')
    pd_items = pd['keyphrase'].split('<peos>')

    if td_items[0] == pd_items[0]:
        target_res[td['asp']][0] += 1
        target_bingo += 1
    if len(pd_items) > 1 and td_items[1] == pd_items[1]:
        opinion_res[td['asp']][0] += 1
        opinion_bingo += 1
    if len(pd_items) > 1 and (td_items[1] == pd_items[1] or ws_tool.similarity(td_items[1], pd_items[1]) == 1):
        opinion_sim_res[td['asp']][0] += 1
        opinion_sim_bingo += 1

    if td_items[0] == pd_items[0] and len(pd_items) > 1 and (
            td_items[1] == pd_items[1] or ws_tool.similarity(td_items[1], pd_items[1]) == 1):
        kp_sim_res[td['asp']][0] += 1
        kp_sim_bingo += 1

    if td_items[0] == pd_items[0] and len(pd_items) > 1 and \
            td_items[1] == pd_items[1] and td['sentiment'] == pd['sentiment']:
        triplet_res[td['asp']][0] += 1
        triplet_bingo += 1

    if td_items[0] == pd_items[0] and len(pd_items) > 1 and (
            td_items[1] == pd_items[1] or ws_tool.similarity(td_items[1], pd_items[1]) == 1) and \
            td['sentiment'] == pd['sentiment']:
        triplet_sim_res[td['asp']][0] += 1
        triplet_sim_bingo += 1

    target_res[td['asp']][1] += 1
    opinion_res[td['asp']][1] += 1
    opinion_sim_res[td['asp']][1] += 1
    kp_sim_res[td['asp']][1] += 1
    triplet_res[td['asp']][1] += 1
    triplet_sim_res[td['asp']][1] += 1

print('sentiment')
for idx, tag in common.idx2asp.items():
    print('tag', tag, f1_score(true_sentiment[idx], pre_sentiment[idx], average='macro'))
print('tag', 'all', f1_score(all_true_sentiment, all_pre_sentiment, average='macro'))
print('tag', 'acc', accuracy_score(all_true_sentiment, all_pre_sentiment))
print()

print('keyphrase')
for idx, tag in common.idx2asp.items():
    print('tag', tag, kp_res[idx][0] / kp_res[idx][1])
print('tag', 'all', bingo / len(true_data))
print()

print('keyphrase sim')
for idx, tag in common.idx2asp.items():
    print('tag', tag, kp_sim_res[idx][0] / kp_sim_res[idx][1])
print('tag', 'all', kp_sim_bingo / len(true_data))
print()

print('triplet')
for idx, tag in common.idx2asp.items():
    print('tag', tag, triplet_res[idx][0] / triplet_res[idx][1])
print('tag', 'all', triplet_bingo / len(true_data))
print()

print('triplet sim')
for idx, tag in common.idx2asp.items():
    print('tag', tag, triplet_sim_res[idx][0] / triplet_sim_res[idx][1])
print('tag', 'all', triplet_sim_bingo / len(true_data))
print()

print('target')
for idx, tag in common.idx2asp.items():
    print('tag', tag, target_res[idx][0] / target_res[idx][1])
print('tag', 'all', target_bingo / len(true_data))
print()

print('opinion')
for idx, tag in common.idx2asp.items():
    print('tag', tag, opinion_res[idx][0] / opinion_res[idx][1])
print('tag', 'all', opinion_bingo / len(true_data))

print('opinion sim')
for idx, tag in common.idx2asp.items():
    print('tag', tag, opinion_sim_res[idx][0] / opinion_sim_res[idx][1])
print('tag', 'all', opinion_sim_bingo / len(true_data))
