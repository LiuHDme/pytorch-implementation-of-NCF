import torch
import numpy as np
import heapq
import math

# Global variables
_model = None
_testList = None
_testNegList = None
_topk = None

def evaluate_model(model, docDataset, K):
    global _model
    global _testList
    global _testNegList
    global _topk
    _model = model
    _testList = docDataset.testList
    _testNegList = docDataset.testNegList
    _topk = K

    hits, ndcgs = [], []
    for idx in range(len(_testList)):
        (hr, ndcg) = eval_one_rating(model, idx, _testList, _testNegList, K)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)

def eval_one_rating(model, idx, testList, testNegList, K):
    u_i_pair = testList[idx]
    items = testNegList[idx]
    u = u_i_pair[0]
    gtItem = u_i_pair[1]
    items.append(gtItem)
    user_input = np.full(len(items), u, dtype='int32')
    item_input = (np.array(items))

    feed_dict = {
        'user_input': user_input,
        'item_input': item_input
    }
    outputs = _model.predict(feed_dict)

    map_item_score = {}
    for i in range(len(items)):
        map_item_score[items[i]] = outputs[i]
    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    hr = getHitRitio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRitio(ranklist, item):
    for i in ranklist:
        if i == item:
            return 1
    return 0

def getNDCG(ranklist, item):
    for i in range(len(ranklist)):
        if ranklist[i] == item:
            return math.log(2) / math.log(i+2)
    return 0
