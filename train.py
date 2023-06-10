import torch
from GPUtil import GPUtil

from utils import load_dataset, make_data, MyDataSet
from torch.utils.data import DataLoader
from pytorch_lighting_model.lighting_tra import model_encoder
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl

if __name__ == '__main__':
    data_load = load_dataset("/mnt/sde/ycl/Nanopore_program_copy/Nanopore_data/small_train.csv")
    # data_load_val = load_dataset("./dataset.pt")
    sequence, nano_data, label = make_data(data_load)
    # enc_inputs_valid, contact_map_valid = make_data(data_load_val)
    mnist_train = MyDataSet(sequence, nano_data, label)
    # [mnist_train, mnist_val] = torch.load("./Example_dataset.pt")
    #proportions = [.8, .1, .1]
    #lengths = [int(p * len(dataset)) for p in proportions]
    #lengths[-1] = len(dataset) - sum(lengths[:-1])
    #mnist_train, mnist_val, mnist_test = random_split(dataset, lengths)
    data_load = load_dataset("/mnt/sde/ycl/Nanopore_program_copy/Nanopore_data/small_test.csv")
    sequence, nano_data, label = make_data(data_load)
    mnist_val = MyDataSet(sequence, nano_data, label)
    # torch.save([mnist_train, mnist_val], "./Example_dataset.pt")

    # add test data
    # data_load_test = load_dataset("/mnt/sde/ycl/Nanopore_program/Nanopore_data/testing_set.csv")
    # sequence, nano_data, label = make_data(data_load_test)
    # test_dataset = MyDataSet(sequence, nano_data, label)

    # [mnist_train, mnist_val] = torch.load("./Example_dataset.pt')
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_AUPRC",
        filename="noContrast-mnist-{epoch:02d}-{avg_val_ACC:.5f}-{avg_val_AUPRC:.5f}-{avg_val_AUROC:.5f}-{avg_val_Precision:.5f}-{avg_val_Recall:.5f}-{avg_val_F1Score:.5f}",
        save_top_k=1,
        mode="max",
        save_last=False
    )
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
    # logger = TensorBoardLogger("tb_logs", name="my_model")
    loader = DataLoader(mnist_train, 2048, True)
    loader_valid = DataLoader(mnist_val, 2048, False)
    #loader_test = DataLoader(mnist_test, 516, True) #232
    model = model_encoder()
    # trainer = pl.Trainer(accelerator="gpu", gpus=[0], limit_train_batches=0.5, max_epochs=500, min_epochs=499,)
    # trainer = pl.Trainer(accelerator="gpu",enable_checkpointing=False, gpus=[0], limit_train_batches=0.5, max_epochs=3, min_epochs=2)
    trainer = pl.Trainer(accelerator="gpu", gpus=gpu, max_epochs=100, min_epochs=90, callbacks=[checkpoint_callback])
    # trainer = pl.Trainer(accelerator="gpu", gpus=[3], limit_train_batches=0.5, max_epochs=2100, min_epochs=2000, callbacks=[checkpoint_callback], resume_from_checkpoint='/mnt/sdb/home/jy/contact_map/lightning_logs/version_25/checkpoints/sample-mnist-epoch=780-val_loss=2.14.ckpt')
    # trainer.fit(model, loader, loader_valid)

    # trainer = pl.Trainer(accelerator="gpu", gpus=[1], max_epochs=1100, min_epochs=1000
    trainer.fit(model, loader, loader_valid)
    trainer.test(model, loader_valid)
    # trainer.save_checkpoint("lightning_logs/example.ckpt")
