import pdb
import random
import string
from timeit import default_timer as timer

import numpy as np

import torch
import torch.nn.functional as F

batches = 100

# idxes = []
#
# begin_gen = timer()
# for i in range(batches):
#     idxes.append(torch.from_numpy(np.random.randint(500000, size=(32, 500))))
# end_gen = timer()
# print('Idx gen time:', end_gen - begin_gen)
#
# embedding_w = torch.randn(500000, 300)
#
# begin_emb = timer()
# embs = []
# for i in range(batches):
#     embs.append(F.embedding(idxes[i], embedding_w))
# end_emb = timer()
# print('Emb time:', end_emb - begin_emb)
#
# begin_cuda = timer()
# for i in range(batches):
#     embs[i].cuda()
# end_cuda = timer()
# print('Cuda time:', end_cuda - begin_cuda)

# itos = []
# stoi = {}
# begin_gen = timer()
# for i in range(100000):
#     s = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
#     stoi[s] = len(itos)
#     itos.append(s)
#
# data_batches = []
# for i in range(batches):
#     idxes = torch.from_numpy(np.random.randint(500000, size=(32, 500)))
#     batch = []
#     for b in range(32):
#         seq = []
#         for w in range(500):
#             seq.append(itos[w])
#         batch.append(seq)
#     data_batches.append(batch)
#
# end_gen = timer()
# print('Idx gen time:', end_gen - begin_gen)
#
# begin_num = timer()
# for i in range(batches):
#     arr = [[stoi[x] for x in ex] for ex in data_batches[i]]
#     arr = torch.LongTensor(arr)
# end_num = timer()
# print('Numericalize time:', end_num - begin_num)

# def a():
#     return 1, 2, 3
#
#
# b = a()
# print(b)

# class A:
#     pass
#
#
# class B(A):
#     pass

# begin_gen = timer()
# objs = []
# for i in range(100):
#     objs.append(B())
# end_gen = timer()
# print('Idx gen time:', end_gen - begin_gen)
#
# begin_op = timer()
# for i in range(100):
#     isinstance(objs[i], A)
# end_op = timer()
# print('Op time:', end_op - begin_op)
#
# begin_op = timer()
# for i in range(100):
#     isinstance(objs[i], B)
# end_op = timer()
# print('Op time:', end_op - begin_op)

# def test(a, b=0, c=0):
#     print(a, b, c)
#
#
# test(1, c=None)

def func(a, b, c):
    print('test')
