# -*- coding: utf-8 -*-
# @Time    : 2023/3/26 21:05
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: utils_nanopore.py

import torch
import torch.utils.data as Data


def load_dataset(filePath, mask=-1):
    import csv
    # data_load = []
    sequence = []
    nano_data = []
    label = []

    with open(filePath, encoding='utf-8-sig') as f:
        csvreader = csv.reader(f, skipinitialspace=True)
        next(csvreader)

        for row in csvreader:
            b = []
            if mask == 3:
                sequence.append('C'*15)
            else:
                sequence.append('C'*2+row[0]+'C'*2)
            signal_means = [float(i) for i in row[1].split(",")]
            signal_stds = [float(i) for i in row[2].split(",")]
            signal_lens = [float(i) for i in row[3].split(",")]
            if mask == 0:
                signal_means = [0]*len(signal_means)
            elif mask == 1:
                signal_stds = [0]*len(signal_stds)
                # signal_lens = [float(i) for i in row[3].split(",")]
            elif mask == 2:
                signal_lens = [0]*len(signal_stds)
            for i, _ in enumerate(signal_means):
                a = []
                a.append(signal_means[i])
                a.append(signal_stds[i])
                a.append(signal_lens[i])
                b.append(a)
            nano_data.append(b)
            label.append(int(row[4]))

    f.close()
    data_load = [sequence, nano_data, label]

    # print(len(data_load))
    return data_load
#
#
# def load_dataset(filePath):
#     import csv
#     # data_load = []
#     sequence = []
#     nano_data = []
#     label = []
#
#     with open(filePath, encoding='utf-8-sig') as f:
#         csvreader = csv.reader(f, skipinitialspace=True)
#         next(csvreader)
#
#         for row in csvreader:
#             b = []
#             sequence.append(row[0][-1]*2+row[0]+row[0][-1]*2)
#             signal_means = [float(i) for i in row[1].split(",")]
#             signal_stds = [float(i) for i in row[2].split(",")]
#             signal_lens = [float(i) for i in row[3].split(",")]
#             for i, _ in enumerate(signal_means):
#                 a = []
#                 a.append(signal_means[i])
#                 a.append(signal_stds[i])
#                 a.append(signal_lens[i])
#                 b.append(a)
#             nano_data.append(b)
#             label.append(int(row[4]))
#
#     f.close()
#     data_load = [sequence, nano_data, label]
#
#     # print(len(data_load))
#     return data_load

def load_dataset1(filePath):
    import csv
    # data_load = []
    sequence = []
    nano_data = []
    label = []

    with open(filePath, encoding='utf-8-sig') as f:
        csvreader = csv.reader(f, skipinitialspace=True)
        next(csvreader)

        for row in csvreader:
            b = []
            sequence.append(row[0])
            signal_means = [float(i) for i in row[1].split(",")]
            signal_stds = [float(i) for i in row[2].split(",")]
            signal_lens = [float(i) for i in row[3].split(",")]
            for i, _ in enumerate(signal_means):
                a = []
                a.append(signal_means[i])
                a.append(signal_stds[i])
                a.append(signal_lens[i])
                b.append(a)
            nano_data.append(b)
            label.append(int(row[4]))

    f.close()
    data_load = [sequence, nano_data, label]

    # print(len(data_load))
    return data_load


tran_code = {
    "PAD": 0, "A": 1, 'C': 2, 'D': 3, 'E': 4,
    'F': 5, 'G': 6, 'H': 7, 'K': 8,
    'I': 9, 'L': 10, 'M': 11, 'N': 12,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16,
    'T': 17, 'V': 18, 'Y': 19, 'W': 20, "X": 21
}

sentences = [
    # 德语和英语的单词个数不要求相同
    # enc_input                dec_input           dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# 德语和英语的单词要分开建立词库
# Padding Should be Zero
src_vocab = {
    "PAD": 0, "A": 1, 'C': 2, 'D': 3, 'E': 4,
    'F': 5, 'G': 6, 'H': 7, 'K': 8,
    'I': 9, 'L': 10, 'M': 11, 'N': 12,
    'P': 13, 'Q': 14, 'R': 15, 'S': 16,
    'T': 17, 'V': 18, 'Y': 19, 'W': 20, "X": 21
}

src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)


def make_data(data_load):
    sequence, nano_data, label = data_load[0], data_load[1], data_load[2]

    label = torch.LongTensor(label)
    nano_data = torch.Tensor(nano_data).float()
    return sequence, nano_data, label


class MyDataSet(Data.Dataset):
    """自定义DataLoader"""

    def __init__(self, seq, nano, label):
        super(MyDataSet, self).__init__()
        self.seq = seq
        self.nano = nano
        self.label = label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return self.seq[idx], self.nano[idx], self.label[idx]
