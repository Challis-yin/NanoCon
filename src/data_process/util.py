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
            sequence.append('C'*2+row[0]+'C'*2)
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

def load_pretrain_data(filePath):
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
                a = [signal_means[i], signal_stds[i], signal_lens[i]]
                b.append(a)
            nano_data.append(b)
            label.append(int(row[4]))

    f.close()
    data_load = [sequence, nano_data, label]

    # print(len(data_load))
    return data_load

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
