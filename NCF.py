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
    parser = argparse.ArgumentParser(description='Run NCF.')
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default='256',
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[16, 32, 16, 8]',
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

class NCF(nn.Module):
    def __init__(self, num_users, num_items, layers):
        super().__init__()
        embed_dim = int(layers[0]/2)
        self.GMF_user_embedding = nn.Embedding(num_users, embed_dim)
        self.GMF_item_embedding = nn.Embedding(num_items, embed_dim)
        self.MLP_user_embedding = nn.Embedding(num_users, embed_dim)
        self.MLP_item_embedding = nn.Embedding(num_items, embed_dim)
        self.fc_layers = nn.ModuleList()
        for (input_shape, output_shape) in zip(layers[:-1], layers[1:]):
            self.fc_layers.append(nn.Linear(input_shape, output_shape))
        self.output_layer = nn.Linear(embed_dim+layers[-1], 1)

    def forward(self, feeddict):
        user_input = feeddict['user_input']
        item_input = feeddict['item_input']
        # GMF
        GMF_user_embedding = self.GMF_user_embedding(user_input)
        GMF_item_embedding = self.GMF_item_embedding(item_input)
        x1 = GMF_user_embedding * GMF_item_embedding
        # MLP
        MLP_user_embedding = self.MLP_user_embedding(user_input)
        MLP_item_embedding = self.MLP_item_embedding(item_input)
        x2 = torch.cat([MLP_user_embedding, MLP_item_embedding], 1)
        for fc_layer in self.fc_layers:
            x2 = fc_layer(x2)
            x2 = F.relu(x2)
        # NCF
        x = torch.cat([x1, x2], 1)
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
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_neg = args.num_neg
    lr = args.lr
    learner = args.learner
    verbose = args.verbose
    out = args.out

    topK = 10
    print('NCF 参数: {} {}'.format(args, '\n'))
    model_out_file = 'Pretrain/model_{}.ckpt'.format(time())

    # Load data
    t1 = time()
    docDataset = DocDataset(path, num_neg)
    train_loader = torch.utils.data.DataLoader(docDataset, batch_size=batch_size, shuffle=True)
    print('数据读取完毕 [{:.1f} s]. #user={}, #item={}, #train={}, #test={} {}'.
        format(time()-t1, docDataset.num_users, docDataset.num_items, docDataset.ratingMatrix.nnz, len(docDataset.testList), '\n'))

    # Build model
    model = NCF(docDataset.num_users, docDataset.num_items, layers).to(device)
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
        print('NCF模型存储于 {}'.format(model_out_file))
