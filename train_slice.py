import numpy as np
import torch
from GPUtil import GPUtil

from utils import load_dataset, make_data, MyDataSet
from torch.utils.data import DataLoader
from pytorch_lighting_model.lighting_slice import model_encoder
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl

def collate(batch):
    seq1_ls=[]
    seq2_ls=[]
    nano1_ls = []
    nano2_ls = []
    label1_ls=[]
    label2_ls=[]
    label_ls=[]
    batch_size=len(batch)
    for i in range(int(batch_size/2)):
        seq1,nano1,label1=batch[i][0], batch[i][1], batch[i][2]
        seq2,nano2,label2=batch[i+int(batch_size/2)][0],batch[i+int(batch_size/2)][1], batch[i+int(batch_size/2)][2]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label=(label1^label2)
        nano1_ls.append(nano1.unsqueeze(0))
        nano2_ls.append(nano2.unsqueeze(0))
        seq1_ls.append(seq1)
        seq2_ls.append(seq2)
        label_ls.append(label.unsqueeze(0))
    nano1=torch.cat(nano1_ls)
    nano2=torch.cat(nano2_ls)
    seq1 = seq1_ls
    seq2 = seq2_ls
    label=torch.cat(label_ls)
    label1=torch.cat(label1_ls)
    label2=torch.cat(label2_ls)
    return seq1,seq2,nano1,nano2,label,label1,label2


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="这是一个argparse示例")
    parser.add_argument("-m", "--mark", help="mask掉的nano维度", default="-1")
    parser.add_argument("-s", "--seed", help="随机种子", default="42")
    args = parser.parse_args()
    print(args)
    #data_load = load_dataset("/mnt/sde/ycl/Nanopore_bert/Nanopore_data/small_5mc.csv", args.mark)
    #data_load = load_dataset("/mnt/sde/ycl/Nanopore_program_copy/Nanopore_data/cd_train.csv", args.mark)
    data_load = load_dataset("/mnt/sde/ycl/Nanopore_program_copy/Nanopore_data/small_train.csv", args.mark)
    # data_load_val = load_dataset("./dataset.pt")
    sequence, nano_data, label = make_data(data_load)
    # enc_inputs_valid, contact_map_valid = make_data(data_load_val)
    mnist_train = MyDataSet(sequence, nano_data, label)
    # [mnist_train, mnist_val] = torch.load("./Example_dataset.pt")
    # proportions = [.9, .1]
    # lengths = [int(p * len(dataset)) for p in proportions]
    # lengths[-1] = len(dataset) - sum(lengths[:-1])
    # mnist_train, mnist_val = random_split(dataset, lengths)
    #proportions = [.99, .1]
    #lengths = [int(p * len(dataset)) for p in proportions]
    #lengths[-1] = len(dataset) - sum(lengths[:-1])
    #mnist_train, mnist_val = random_split(dataset, lengths)
    #data_load = load_dataset("/mnt/sde/ycl/Nanopore_program_copy/Nanopore_data/cd_test.csv", args.mark)
    data_load = load_dataset("/mnt/sde/ycl/Nanopore_program_copy/Nanopore_data/small_test.csv", args.mark)
    # data_load_val = load_dataset("./dataset.pt")
    sequence, nano_data, label = make_data(data_load)
                # enc_inputs_valid, contact_map_valid = make_data(data_load_val)
    mnist_val = MyDataSet(sequence, nano_data, label)

    # torch.save([mnist_train, mnist_val], "./Example_dataset.pt")

    # add test data
    # data_load_test = load_dataset("/mnt/sdb/home/wrh/Nanopore_program/Nanopore_data/testing_set.csv")
    # sequence, nano_data, label = make_data(data_load_test)
    # test_dataset = MyDataSet(sequence, nano_data, label)

    # [mnist_train, mnist_val] = torch.load("./Example_dataset.pt")
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_AUPRC",
        filename="sample-mnist-{epoch:02d}-{avg_val_ACC:.5f}-{avg_val_AUPRC:.5f}-{avg_val_AUROC:.5f}-{avg_val_Precision:.5f}-{avg_val_Recall:.5f}-{avg_val_F1Score:.5f}",
        save_top_k=1,
        mode="max",
        save_last=False
    )

    loader = DataLoader(mnist_train, 512, True, collate_fn=collate, num_workers=12)
    loader_valid = DataLoader(mnist_val, 512, False, collate_fn=collate)

    flag = True
    gpu = []
    while flag:
        # 获取所有可用的GPU
        gs = GPUtil.getGPUs()
        # 遍历所有GPU并检查它们的显存大小
        for i, g in enumerate(gs):
            if g.memoryFree > 10000:
                flag = False
                gpu.append(i)
                break

    # loader_test = DataLoader(mnist_test, 2048, False, collate_fn=collate) #232
    model = model_encoder()
    # trainer = pl.Trainer(accelerator="gpu", gpus=[0], limit_train_batches=0.5, max_epochs=500, min_epochs=499,)
    # trainer = pl.Trainer(accelerator="gpu",enable_checkpointing=False, gpus=[0], limit_train_batches=0.5, max_epochs=3, min_epochs=2)
    # seed = args.seed
    # # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    trainer = pl.Trainer(accelerator="gpu", gpus=gpu, max_epochs=50, min_epochs=40, callbacks=[checkpoint_callback])
    # trainer = pl.Trainer(accelerator="gpu", gpus=[2],  max_epochs=50, min_epochs=40, callbacks=[checkpoint_callback], resume_from_checkpoint='/mnt/sde/ycl/Nanopore_bert/lightning_logs/version_28/checkpoints/sample-mnist-epoch=33-avg_val_ACC=0.99-avg_val_AUPRC=0.83-avg_val_AUROC=0.99-avg_val_Precision=0.83-avg_val_Recall=0.69-avg_val_F1Score=0.75.ckpt')
    # trainer.fit(model, loader, loader_valid)
    trainer.fit(model, loader, loader_valid)
    trainer.test(model, loader_valid)
    # trainer.save_checkpoint("lightning_logs/example.ckpt")
