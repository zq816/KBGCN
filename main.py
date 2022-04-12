import argparse
import numpy as np
import tensorflow as tf
from time import time
from data_loader import load_data
from train import train

#np.random.seed(555)


parser = argparse.ArgumentParser()

'''
# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
'''


'''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
'''



# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=15, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')



show_loss = False
show_time = True
show_topk = False

t = time()

args = parser.parse_args()
data = load_data(args)
#train(args, data, show_loss, show_topk)

for i in range(100):
    tf.reset_default_graph()
    train(args, data, show_loss, show_topk)
    writer = open('../src/' +  'music_100.txt', 'a', encoding='utf-8')
    resauc = []
    resf1 = []
    for line in open('../src/' + 'result.txt', encoding='utf-8'):
        resauc.append(list(map(float,line.split('    ')))[0])
        resf1.append(list(map(float,line.split('    ')))[1])
    resauc.sort()
    resf1.sort()
    writer.write('%.4f    %.4f\n' % (resauc[14],resf1[14]))
    print('%.4f    %.4f' % (resauc[14],resf1[14]))


if show_time:
    print('time used: %d s' % (time() - t))
