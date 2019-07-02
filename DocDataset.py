import scipy.sparse as sp
import numpy as np
import torch.utils.data

class DocDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_neg=5):
        # Rating matrix
        self.ratingMatrix = self.load_rating_file_as_matrix(path+'movielens.train.rating')
        self.num_users, self.num_items = self.ratingMatrix.shape
        # Training set
        self.user_input, self.item_input, self.labels = self.get_train_instances(self.ratingMatrix, num_neg)
        # Test set
        self.testList = self.load_rating_file_as_list(path+'movielens.test.rating')
        self.testNegList = self.create_neg_list(self.testList)

    def __len__(self):
        return len(self.user_input)

    def __getitem__(self, index):
        'Generates one sample of data.'
        user_input = self.user_input[index]
        item_input = self.item_input[index]
        label = self.labels[index]
        return {
            'user_input': user_input,
            'item_input': item_input,
            'label': label
        }

    def get_train_instances(self, ratingMatrix, num_neg):
        user_input, item_input, labels = [], [], []
        for (u, i) in ratingMatrix.keys():
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            for _ in range(num_neg):
                j = np.random.randint(self.num_items)
                while (u, j) in ratingMatrix:
                    j = np.random.randint(self.num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return user_input, item_input, labels

    def load_rating_file_as_matrix(self, filename):
        '''
        读取 ratings.csv，并返回 matrix
        '''
        num_users, num_items = 0, 0
        with open(filename, 'r') as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        with open(filename, 'r') as f:
            line = f.readline()
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                u, i = int(arr[0]), int(arr[1])
                mat[u-1, i-1] = 1.0
                line = f.readline()
        return mat

    def load_rating_file_as_list(self, filename):
        '''
        读取 ratings.csv，并返回 list
        '''
        list = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.split('\t')
                u, i = int(arr[0]), int(arr[1])
                list.append([u-1, i-1])
                line = f.readline()
        return list

    def create_neg_list(self, testList, num_neg=100):
        '''
        对于 testList 中的每一个 user-item pair，产生 num_neg 个负样本
        '''
        neg_list = []
        num_items = self.ratingMatrix.shape[1]
        for u_i_pair in testList:
            u = u_i_pair[0]
            i = u_i_pair[1]
            neg = []
            for _ in range(num_neg):
                j = np.random.randint(num_items)
                while (u, j) in self.ratingMatrix or j == i:
                    j = np.random.randint(num_items)
                neg.append(j)
            neg_list.append(neg)
        return neg_list
