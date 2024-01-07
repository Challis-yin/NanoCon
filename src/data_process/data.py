# -*- coding: utf-8 -*-
# @Time    : 2023/12/12 10:01
# @Author  : Yin Cheng lin
# @Email   : sdyinruichao@163.com
# @IDE     : PyCharm

import torch
import torch.utils.data as Data
import numpy as np
import random
import lmdb


def load_pretrain_data(filePath):
    import csv
    # data_load = []
    sequences = []
    stats = []
    signal = []

    with open(filePath, encoding='utf-8-sig') as f:
        csvreader = csv.reader(f, skipinitialspace=True)
        next(csvreader)
        for row in csvreader:
            b = []
            signal_i = []
            sequences.append(row[0])
            signal_means = [float(i) for i in row[1].split(",")]
            signal_stds = [float(i) for i in row[2].split(",")]
            signal_lens = [float(i) for i in row[3].split(",")]
            for i, _ in enumerate(signal_means):
                a = [signal_means[i], signal_stds[i], signal_lens[i]]
                b.append(a)
            stats.append(b)
            signal_rows = row[4].split(';')
            for s_r in signal_rows:
                col = s_r.split(',')
                row_array = [float(i) for i in col]
                signal_i.append(row_array)
            signal.append(signal_i)
    f.close()
    data_load = [sequences, signal, stats]
    return data_load

def make_data(data_load):
    sequences, signal, stats = data_load[0], data_load[1], data_load[2]
    signal = torch.Tensor(signal).float()
    stats = torch.Tensor(stats).float()
    return sequences, signal, stats

class MultiModalDataset(Data.Dataset):
    def __init__(self, root):
        self.root = root

        self.env = lmdb.open(self.root,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin()

        self.length = int(self.txn.get(b'__len__').decode())


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # sequence = self.sequences[idx]
        # signal = self.signals[idx]
        # stat = self.stats[idx]
        #
        # # Apply masking to the sequence
        # masked_sequence, mask = self.mask_sequence(sequence)
        #
        # # Convert sequence to tensor
        # masked_sequence_tensor = torch.tensor([self.char_to_int(c) for c in masked_sequence], dtype=torch.long)
        #
        # # Get only the masked original bases
        # original_bases = torch.tensor([self.char_to_int(sequence[i]) for i in range(len(sequence)) if mask[i]], dtype=torch.long)
        #
        # return masked_sequence_tensor, torch.tensor(signal, dtype=torch.float), torch.tensor(stat, dtype=torch.float), original_bases
        data = self.txn.get(str(idx).encode())
        data = data.decode()
        data = data.split('\t')

        sequence = data[0]
        signal = []
        stat = []

        signal_means = [float(i) for i in data[1].split(",")]
        signal_stds = [float(i) for i in data[2].split(",")]
        signal_lens = [float(i) for i in data[3].split(",")]
        for i, _ in enumerate(signal_means):
            a = [signal_means[i], signal_stds[i], signal_lens[i]]
            stat.append(a)
        signal_rows = data[4].split(';')
        for s_r in signal_rows:
            col = s_r.split(',')
            row_array = [float(i) for i in col]
            signal.append(row_array)

        # Apply masking to the sequence
        masked_sequence, mask = self.mask_sequence(sequence)

        # Convert sequence to tensor
        masked_sequence_tensor = torch.tensor([self.char_to_int(c) for c in masked_sequence], dtype=torch.long)

        # Get only the masked original bases
        original_bases = torch.tensor([self.char_to_int(sequence[i]) for i in range(len(sequence)) if mask[i]], dtype=torch.long)

        return masked_sequence_tensor, torch.tensor(signal, dtype=torch.float), torch.tensor(stat, dtype=torch.float), original_bases

        return data


    def mask_sequence(self, sequence, mask_prob=0.15, mask_token='N'):
        chars = list(sequence)
        mask = [False] * len(chars)

        mid_index = len(chars) // 2
        mask_index = random.choice([i for i in range(len(chars)) if i != mid_index])
        chars[mask_index] = mask_token
        mask[mask_index] = True

        return ''.join(chars), mask

    @staticmethod
    def char_to_int(char):
        char_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        return char_dict.get(char, 4)

if __name__ == '__main__':
    data_load = load_pretrain_data("/mnt/sde/ycl/NanoCon/code/pretrain/nanopore_data/test.csv")
    print("successful")
    # # 示例数据
    # sequences = ["AGCTCGTACGTCG", "GTCAGTCGACGAT", "TCGATCGTAGCTA"]  # 示例序列
    # signals = [np.random.rand(16, 13) for _ in sequences]  # 随机生成的电信号数据
    # stats = [np.random.rand(3, 13) for _ in sequences]     # 随机生成的统计数据
    #
    # # 创建数据集实例
    # dataset = MultiModalDataset(sequences, signals, stats)