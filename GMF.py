# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# Workspace imports
from DocDataset import DocDataset
from evaluate import evaluate_model
# Python imports
import argparse
import numpy as np
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Run GMF.')
    parser.add_argument('--path', nargs='?', default='/Users/liuhdme/Documents/推荐系统/learnRS/recommender_pytorch/Data/',
                        help='Input data path.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default='256',
                        help='Batch size.')
    parser.add_argument('--embed_dim', type=int, default='8',
                        help='Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.')
    parser.add_argument('--reg_layers', nargs='?', default='[0.0, 0.0, 0.0, 0.0]',
                        help='Regularization for each layer.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

class GMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, feeddict):
        user_input = feeddict['user_input']
        item_input = feeddict['item_input']
        user_embedding = self.user_embedding(user_input)
        item_embedding = self.item_embedding(item_input)
        x = user_embedding * item_embedding
        logit = self.output_layer(x)
        output = torch.sigmoid(logit)
        return output

    def predict(self, feed_dict):
        for key in feed_dict:
            feed_dict[key] = torch.from_numpy(feed_dict[key]).to(dtype=torch.long, device=device)
        outputs = self.forward(feed_dict)
        return outputs.cpu().detach().numpy()

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    epochs = args.epochs
    batch_size = args.batch_size
    embed_dim = args.embed_dim
    reg_layers = eval(args.reg_layers)
    num_neg = args.num_neg
    lr = args.lr
    learner = args.learner
    verbose = args.verbose
    out = args.out

    topK = 10
    print('GMF 参数: {} {}'.format(args, '\n'))
    model_out_file = 'Pretrain/GMFmodel_{}.ckpt'.format(time())

    # Load data
    t1 = time()
    docDataset = DocDataset(path, num_neg)
    train_loader = torch.utils.data.DataLoader(docDataset, batch_size=batch_size, shuffle=True)
    print('数据读取完毕 [{:.1f} s]. #user={}, #item={}, #train={}, #test={} {}'.
        format(time()-t1, docDataset.num_users, docDataset.num_items, docDataset.ratingMatrix.nnz, len(docDataset.testList), '\n'))

    # Build model
    model = GMF(docDataset.num_users, docDataset.num_items, embed_dim).to(device)
    print(model)
    print('\n')
    # Loss and optimizer
    criterion = nn.BCELoss()
    if learner.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif learner.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif learner.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Check init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, docDataset, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('初始模型: HR = {:.4f}, NDCG = {:.4f} [{:.1f} s] {}'.
        format(hr, ndcg, time()-t1, '\n'))

    # Train
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Training
        for feed_dict in train_loader:
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
        if epoch % verbose == 0:
            t2 = time()
            (hits, ndcgs) = evaluate_model(model, docDataset, topK)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('Epoch {} [{:.1f} s]: HR = {:.4f}, NDCG = {:.4f}, loss = {:.4f} [{:.1f} s]'.
                    format(epoch, t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if out > 0:
                    torch.save(model.state_dict(), model_out_file)
        # Negative sample again
        docDataset.user_input, docDataset.item_input, docDataset.labels = \
            docDataset.get_train_instances(docDataset.ratingMatrix, num_neg=num_neg)
        train_loader = torch.utils.data.DataLoader(docDataset, batch_size=batch_size, shuffle=True)

    print('训练结束. Best Epoch {}: HR = {:.4f}, NDCG = {:.4f} {}'.
        format(best_iter, best_hr, best_ndcg, '\n'))
    if out > 0:
        print('GMF模型存储于 {}'.format(model_out_file))
