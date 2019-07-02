from DocDataset import DocDataset
from evaluate import evaluate_model

import argparse
import numpy as np
from time import time

def parse_args():
    parser = argparse.ArgumentParser(description='Run ItmePop')
    parser.add_argument('--path', nargs='?', default='/Users/liuhdme/Documents/推荐系统/learnRS/recommender_pytorch/Data/', help='Input data path.')
    return parser.parse_args()

class Itmepop():
    def __init__(self, trainMatrix):
        self.itemsRatingsCountsList = np.array(trainMatrix.sum(axis=0, dtype=int)).flatten()

    def forward(self):
        pass

    def predict(self, feeddict):
        item_input = feeddict['item_input']
        outputs = [self.itemsRatingsCountsList[item_id] for item_id in item_input]
        return np.array(outputs)

if __name__ == "__main__":
    args = parse_args()
    path = args.path

    topK = 10

    docDataset = DocDataset(path)
    model = Itmepop(docDataset.ratingMatrix)
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, docDataset, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Eval: HR = {:.4f}, NDCG = {:.4f} [{:.1f} s]'.format(hr, ndcg, time()-t1))
