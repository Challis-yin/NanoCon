import argparse
import logging
import numpy as np
import torch
from GPUtil import GPUtil

from utils import load_dataset, make_data, MyDataSet
from torch.utils.data import DataLoader
from pytorch_lighting_model.NanoCon import model_encoder
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

# 自定义的函数和类（例如 load_dataset, make_data, MyDataSet, model_encoder）需要被导入或在这个脚本中定义
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_args(args):
    if args.mode == 'train':
        assert args.train_data, "Training data path is required in train mode"
        assert args.test_data, "Testing data path is required in train mode"
        # assert args.save_model_path, "Model save path is required in train mode"
    elif args.mode == 'test':
        assert args.test_data, "Test data path is required in test mode"
        assert args.checkpoint, "Checkpoint path is required in test mode"

def main(args):
    setup_logging()
    validate_args(args)

    torch.manual_seed(args.seed)

    if args.mode == 'train':
        train_data_load = load_dataset(args.train_data)
        val_data_load = load_dataset(args.test_data)
        # Construct training and validation datasets
        # Example: sequence, nano_data, label = make_data(train_data_load, val_data_load)
        sequence, nano_data, label = make_data(train_data_load)
        sequence_val, nano_data_val, label_val = make_data(val_data_load)
        mnist_train = MyDataSet(sequence, nano_data, label)
        mnist_val = MyDataSet(sequence_val, nano_data_val, label_val)
        # Create DataLoaders for training and validation
        loader = DataLoader(mnist_train, args.batch_size, True, collate_fn=collate)
        loader_valid = DataLoader(mnist_val, args.batch_size, False, collate_fn=collate)

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_model_path,
            monitor="avg_val_AUPRC",
            filename="sample-mnist-{epoch:02d}-{avg_val_ACC:.5f}-{avg_val_AUPRC:.5f}-{avg_val_AUROC:.5f}-{avg_val_Precision:.5f}-{avg_val_Recall:.5f}-{avg_val_F1Score:.5f}",
            save_top_k=1,
            mode="max",
            save_last=True
        )

        gpu = [i for i, g in enumerate(GPUtil.getGPUs()) if g.memoryFree > 10000] if args.use_gpu else []
        model = model_encoder()
        trainer = pl.Trainer(accelerator="gpu" if args.use_gpu else "cpu", gpus=gpu, max_epochs=args.epochs, min_epochs=args.epochs - 10, callbacks=[checkpoint_callback])
        trainer.fit(model, loader, loader_valid)

    elif args.mode == 'test':
        test_data_load = load_dataset(args.test_data, args.mark)
        # Construct the test dataset
        # Example: sequence, nano_data, label = make_data(test_data_load)

        mnist_test = MyDataSet(sequence, nano_data, label)
        # Create DataLoader for testing
        loader_test = DataLoader(mnist_test, args.batch_size, False, collate_fn=collate)

        model = model_encoder()
        model.load_from_checkpoint(args.checkpoint)
        trainer = pl.Trainer(accelerator="gpu" if args.use_gpu else "cpu", gpus=[0] if args.use_gpu else [])
        trainer.test(model, loader_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training and testing a deep learning model")
    parser.add_argument("-s", "--seed", help="Random seed", default="42")
    parser.add_argument("--mode", help="Operating mode: train or test", choices=['train', 'test'], required=True)
    parser.add_argument("--train_data", help="Path to the training dataset", required=False)
    parser.add_argument("--val_data", help="Path to the validation dataset", required=False)
    parser.add_argument("--use_gpu", help="Flag to use GPU if available", action="store_true")
    parser.add_argument("--test_data", help="Path to the testing dataset", required=False)
    parser.add_argument("--checkpoint", help="Path to the model checkpoint", required=False)
    parser.add_argument("--batch_size", help="Batch size for training and testing", type=int, default=512)
    parser.add_argument("--epochs", help="Number of epochs for training", type=int, default=50)
    parser.add_argument("--save_model_path", help="Path to save the trained model", required=False)
    args = parser.parse_args()
    main(args)