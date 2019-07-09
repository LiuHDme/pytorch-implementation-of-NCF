import csv
from evaluate import evaluate_model
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin
from time import time
import numpy as np
from NCF import NCF
import torch.utils.data
import torch.nn as nn
from DocDataset import DocDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_EVALS = 500
N_FOLDS = 10
epochs = 30
out_file = 'trials.csv'
path = 'Data/'
topK = 10
batch_size = 256

def train(model, train_loader, optimizer, criterion, docDataset, num_neg):
    best_hr, best_ndcg, best_iter = 0, 0, -1
    for epoch in range(epochs):
        print('eopch: {}'.format(epoch+1))
        # Training
        for i, feed_dict in enumerate(train_loader):
            if i % 10 == 0:
                print('step: {} / {}'.format(i, len(train_loader)))
            for key in feed_dict:
                if type(feed_dict[key]) != type(None):
                    feed_dict[key] = feed_dict[key].to(dtype = torch.long, device = device)
            # Forward pass
            outputs = model(feed_dict)
            labels = feed_dict['label'].float().view(outputs.shape)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Evaluation
        (hits, ndcgs) = evaluate_model(model, docDataset, topK)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
        # Negative sample again
        docDataset.user_input, docDataset.item_input, docDataset.labels = \
            docDataset.get_train_instances(docDataset.ratingMatrix, num_neg=num_neg)
        train_loader = torch.utils.data.DataLoader(docDataset, batch_size=batch_size, shuffle=True)
    return best_hr, best_ndcg

def objective(params):
    global ITERATION
    ITERATION += 1

    start = time()

    docDataset = DocDataset(path, int(params['num_neg']))
    train_loader = torch.utils.data.DataLoader(docDataset, batch_size=batch_size, shuffle=True)
    layers = [params['embed_dim'], 32, 16, params['factor']]
    model = NCF(docDataset.num_users, docDataset.num_items, layers).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=params['learning_rate'])
    criterion = nn.BCELoss()
    best_hr, best_ndcg = train(model, train_loader, optimizer, criterion, docDataset, int(params['num_neg']))

    run_time = time() - start

    # Extract the best score
    loss = 1 - best_hr

    with open(out_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([loss, best_hr, best_ndcg, params, ITERATION, run_time])

    return {
        'loss': loss,
        'best_hr': best_hr,
        'best_ndcg': best_ndcg,
        'params': params,
        'iteration': ITERATION,
        'train_time': run_time,
        'status': STATUS_OK}

if __name__ == '__main__':
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)),
        'embed_dim': hp.choice('embed_dim', [8, 16]),
        'factor': hp.choice('factor', [16, 8]),
        'num_neg': hp.quniform('num_neg', 1, 10, 1)
    }

    global  ITERATION
    ITERATION = 0

    bayes_trials = Trials()
    # File to save first results
    with open(out_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['loss', 'best_hr', 'best_ndcg', 'params', 'iteration', 'train_time', 'status'])

    best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))

    bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
    bayes_trials_results[:2]
