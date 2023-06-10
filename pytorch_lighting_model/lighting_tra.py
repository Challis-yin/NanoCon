import os
import torch
from torch import nn
import pytorch_lightning as pl
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import T5Model, T5Tokenizer
import re
import math
import torch.nn.functional as F
from torcheval.metrics import BinaryAUPRC, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class model_encoder(pl.LightningModule):
    def __init__(self, model_dim=128):
        super().__init__()
        self.kmer = 5
        self.pretrainpath = '/home/weilab/molecular_analysis_server/biology_python/pretrain/DNAbert_5mer'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath)

        self.embedding = nn.Embedding(num_embeddings=1100, embedding_dim=model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.nano_embedding = nn.Linear(3, 128)
        self.bi_gru = nn.GRU(input_size=model_dim*2,
                             num_layers=2,
                             bidirectional=True,
                             hidden_size=model_dim,
                             batch_first=True)
        self.classification = nn.Sequential(nn.Linear(3840, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 2),
                                    )

    def forward(self, x, nano_data):

        seqs = list(x)
        kmer = [[seqs[i][x:x + self.kmer] for x in range(len(seqs[i]) + 1 - self.kmer)] for i in range(len(seqs))]
        kmers = [" ".join(kmer[i]) for i in range(len(kmer))]
        token_seq = self.tokenizer(kmers, return_tensors='pt', padding=True)
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
        input_ids = input_ids.cuda()
        input_ids = input_ids[:, 1:-1]

        input_ids = self.embedding(input_ids)
        input_ids = self.positional_encoding(input_ids)
        seq_outputs = self.transformer_encoder(input_ids)
        # print("nano_data", nano_data.size())
        nano_outputs = self.nano_embedding(nano_data)
        # print("seq_outputs", seq_outputs.size())
        # print("nano_outputs", nano_outputs.size())
        enc_outputs = torch.cat([seq_outputs, nano_outputs], dim=2)
        enc_outputs, hn = self.bi_gru(enc_outputs)
        output = enc_outputs.reshape(enc_outputs.shape[0], -1)
        hn = hn.reshape(output.shape[0], -1)
        output = torch.cat([output, hn], 1)
        logits = self.classification(output)

        return logits, output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # with torch.autograd.detect_anomaly():
        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = batch

        output, _ = self.forward(x, nano_data)

        # calculate the accuracy
        acc_sum = (output.argmax(dim=1) == y).float().sum().item()
        n = y.shape[0]
        acc = acc_sum / n

        loss = criterion(output, y)
        # loss.backward(retain_graph=True)

        predictions = F.softmax(output, dim=1)
        predictions = predictions[:, 1]
        predictions = predictions.cpu()
        y = y.cpu()
        # print(predictions)
        # print(y)
        # print("11111prediction: ", predictions.size())
        # print("11111y: ", y.size())


        metric1 = BinaryAUPRC()
        metric1.update(predictions, y)
        AUPRC = metric1.compute()

        metric2 = BinaryAUROC()
        metric2.update(predictions, y)
        AUROC = metric2.compute()

        metric3 = BinaryPrecision()
        metric3.update(predictions, y)
        Precision = metric3.compute()

        metric4 = BinaryRecall()
        metric4.update(predictions, y)
        Recall = metric4.compute()

        metric5 = BinaryF1Score()
        metric5.update(predictions, y)
        F1Score = metric5.compute()

        # Logging to TensorBoard by default
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ACC", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_AUPRC", AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_AUROC", AUROC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) #1e-3
        return optimizer

    def validation_step(self, val_batch, batch_idx):
        # self.eval()
        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = val_batch

        output, _ = self.forward(x, nano_data)

        # calculate the accuracy
        acc_sum = (output.argmax(dim=1) == y).float().sum().item()
        n = y.shape[0]
        acc = acc_sum / n

        loss = criterion(output, y)

        predictions = F.softmax(output, dim=1)
        predictions = predictions[:, 1]
        predictions = predictions.cpu()
        y = y.cpu()

        # print("11111prediction: ", predictions.size())
        # print("11111y: ", y.size())

        metric1 = BinaryAUPRC()
        metric1.update(predictions, y)
        AUPRC = metric1.compute()

        metric2 = BinaryAUROC()
        metric2.update(predictions, y)
        AUROC = metric2.compute()

        metric3 = BinaryPrecision()
        metric3.update(predictions, y)
        Precision = metric3.compute()

        metric4 = BinaryRecall()
        metric4.update(predictions, y)
        Recall = metric4.compute()

        metric5 = BinaryF1Score()
        metric5.update(predictions, y)
        F1Score = metric5.compute()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ACC", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_AUPRC", AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_AUROC", AUROC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)
        acc = torch.tensor(acc, dtype=float).cuda()

        # 返回损失值字典
        # print('val_loss:', loss)
        return {'val_loss': loss, 'val_ACC': acc, 'val_AUPRC': AUPRC, 'val_AUROC': AUROC, 'val_Precision': Precision, 'val_Recall': Recall, 'val_F1Score': F1Score,}


    def validation_epoch_end(self, outputs):
        # 计算并输出平均损失
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        avg_ACC = torch.stack([x['val_ACC'] for x in outputs]).mean()
        self.log('avg_val_ACC', avg_ACC, on_step=False, on_epoch=True, prog_bar=True)
        avg_AUPRC = torch.stack([x['val_AUPRC'] for x in outputs]).mean()
        self.log('avg_val_AUPRC', avg_AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        avg_AUROC = torch.stack([x['val_AUROC'] for x in outputs]).mean()
        self.log('avg_val_AUROC', avg_AUROC, on_step=False, on_epoch=True, prog_bar=True)
        avg_Precision = torch.stack([x['val_Precision'] for x in outputs]).mean()
        self.log('avg_val_Precision', avg_Precision, on_step=False, on_epoch=True, prog_bar=True)
        avg_Recall = torch.stack([x['val_Recall'] for x in outputs]).mean()
        self.log('avg_val_Recall', avg_Recall, on_step=False, on_epoch=True, prog_bar=True)
        avg_F1Score = torch.stack([x['val_F1Score'] for x in outputs]).mean()
        self.log('avg_val_F1Score', avg_F1Score, on_step=False, on_epoch=True, prog_bar=True)

    # def on_validation_end(self):
    #     # 强制将验证集损失值写入日志系统
    #     self.log('val_loss', self.trainer.callback_metrics['val_loss'], on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = test_batch

        output, rep = self.forward(x, nano_data)

        # calculate the accuracy
        acc_sum = (output.argmax(dim=1) == y).float().sum().item()
        n = y.shape[0]
        acc = acc_sum / n

        loss = criterion(output, y)

        predictions = F.softmax(output, dim=1)
        predictions = predictions[:, 1]
        predictions = predictions.cpu()
        y = y.cpu()

        metric1 = BinaryAUPRC()
        metric1.update(predictions, y)
        AUPRC = metric1.compute()

        metric2 = BinaryAUROC()
        metric2.update(predictions, y)
        AUROC = metric2.compute()

        metric3 = BinaryPrecision()
        metric3.update(predictions, y)
        Precision = metric3.compute()

        metric4 = BinaryRecall()
        metric4.update(predictions, y)
        Recall = metric4.compute()

        metric5 = BinaryF1Score()
        metric5.update(predictions, y)
        F1Score = metric5.compute()

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_ACC", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_AUPRC", AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_AUROC", AUROC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # 计算并输出平均损失
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        print('avg_test_loss:', avg_loss)

# data
# mnist_train, mnist_val = random_split(dataset, [55000, 5000])
#
# train_loader = DataLoader(mnist_train, batch_size=32)
# val_loader = DataLoader(mnist_val, batch_size=32)
#
# # model
#
# # training
# trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
# trainer.fit(model, train_loader, val_loader)
