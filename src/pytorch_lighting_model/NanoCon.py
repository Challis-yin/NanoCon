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

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception

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

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):


        cos_distance = D(output1, output2)
        # print("ED",euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 3))

        while loss_contrastive<0.1:
            loss_contrastive = loss_contrastive*10
        return loss_contrastive

class model_encoder(pl.LightningModule):
    def __init__(self, model_dim=128):
        super().__init__()
        self.kmer = 5
        self.pretrainpath = '/home/weilab/molecular_analysis_server/biology_python/pretrain/DNAbert_5mer'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath)

        self.learning_rate = 1e-3
        self.weight_decay = 0.01

        self.embedding = nn.Embedding(num_embeddings=1100, embedding_dim=model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.nano_embedding = nn.Linear(3, 128)
        # self.nano_embedding = nn.Sequential(nn.Linear(3, 512),
        #               nn.BatchNorm1d(512),
        #               nn.ReLU(),
        #               nn.Linear(512, 128),
        #               )
        self.bi_gru = nn.GRU(input_size=model_dim*2,
                             num_layers=2,
                             bidirectional=True,    
                             hidden_size=model_dim,
                             batch_first=True)
        self.cls1 = nn.Sequential(nn.Linear(3840, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 128),
                                    )
        self.cls2 = nn.Sequential(
                                  nn.BatchNorm1d(128),
                                  nn.ReLU(),
                                  nn.Linear(128, 1024),
                                  nn.BatchNorm1d(1024),
                                  nn.ReLU(),
                                  nn.Linear(1024, 128),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU(),
                                  nn.Linear(128, 2),
                                  )
        self.cls3 = nn.Sequential(nn.Linear(256, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 128),
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
        nano_data = nano_data.cuda()
        nano_outputs = self.nano_embedding(nano_data)
        # print("seq_outputs", seq_outputs.size())
        # print("nano_outputs", nano_outputs.size())
        enc_outputs = torch.cat([seq_outputs, nano_outputs], dim=2)
        enc_outputs, hn = self.bi_gru(enc_outputs)
        # output = enc_outputs.reshape(enc_outputs.shape[0], -1)
        # hn = hn.reshape(output.shape[0], -1)
        # output = torch.cat([output, hn], 1)
        #repre = self.cls3(torch.transpose(enc_outputs[:, len(enc_outputs[0]), :], 0, 1))
        repre = self.cls3(enc_outputs[:, int((1+len(enc_outputs[0])/2)), :])
        # repre = self.cls1(output)

        return repre

    def get_logits(self, x, nano_data):
        with torch.no_grad():
            repre = self.forward(x, nano_data)
        logits = self.cls2(repre)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # with torch.autograd.detect_anomaly():
        criterion = nn.CrossEntropyLoss()
        contrast_cri = ContrastiveLoss()
        # x, nano_data, y = batch
        seq1, seq2, nano1, nano2, label, label1, label2 = batch

        repre1 = self.forward(seq1, nano1)
        logits1 = self.get_logits(seq1, nano1)
        repre2 = self.forward(seq2, nano2)
        logits2 = self.get_logits(seq2, nano2)

        contrast_loss = contrast_cri(repre1, repre2, label)
        ce_loss1 = criterion(logits1, label1)
        ce_loss2 = criterion(logits2, label2)
        loss = 10*contrast_loss + ce_loss1 + ce_loss2

        # calculate the accuracy
        acc_sum1 = (logits1.argmax(dim=1) == label1).float().sum().item()
        acc_sum2 = (logits2.argmax(dim=1) == label2).float().sum().item()
        n = label1.shape[0] + label2.shape[0]
        acc = (acc_sum1+acc_sum2) / n

        # loss.backward(retain_graph=True)
        output = torch.cat((logits1, logits2), dim=0)
        y = torch.cat((label1, label2), dim=0)
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
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_ACC", acc, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_AUPRC", AUPRC, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_AUROC", AUROC, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_Precision", Precision, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_Recall", Recall, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_F1Score", F1Score, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from pytorch_lightning.callbacks import LearningRateMonitor
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) #1e-3
        # return optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        monitor = "val_AUPRC" # 监视指标为验证集损失
        lr_monitor = LearningRateMonitor(logging_interval='step')
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': monitor,
            'callbacks': [lr_monitor]
        }

    def validation_step(self, val_batch, batch_idx):
        # self.eval()
        criterion = nn.CrossEntropyLoss()
        contrast_cri = ContrastiveLoss()
        # x, nano_data, y = batch
        seq1, seq2, nano1, nano2, label, label1, label2 = val_batch

        repre1 = self.forward(seq1, nano1)
        logits1 = self.get_logits(seq1, nano1)
        repre2 = self.forward(seq2, nano2)
        logits2 = self.get_logits(seq2, nano2)

        contrast_loss = contrast_cri(repre1, repre2, label)
        ce_loss1 = criterion(logits1, label1)
        ce_loss2 = criterion(logits2, label2)
        loss = contrast_loss + ce_loss1 + ce_loss2

        # calculate the accuracy
        acc_sum1 = (logits1.argmax(dim=1) == label1).float().sum().item()
        acc_sum2 = (logits2.argmax(dim=1) == label2).float().sum().item()
        n = label1.shape[0] + label2.shape[0]
        acc = (acc_sum1 + acc_sum2) / n

        # loss.backward(retain_graph=True)
        output = torch.cat((logits1, logits2), dim=0)
        y = torch.cat((label1, label2), dim=0)
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
        # AUPRC = torch.tensor(AUPRC, dtype=float).cuda()
        # AUROC = torch.tensor(AUROC, dtype=float).cuda()
        # Precision = torch.tensor(Precision, dtype=float).cuda()
        # Recall = torch.tensor(Recall, dtype=float).cuda()
        # F1Score = torch.tensor(F1Score, dtype=float).cuda()
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
        # print('avg_val_loss:', avg_loss)

    # def on_validation_end(self):
    #     # 强制将验证集损失值写入日志系统
    #     self.log('val_loss', self.trainer.callback_metrics['val_loss'], on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        criterion = nn.CrossEntropyLoss()
        contrast_cri = ContrastiveLoss()
        # x, nano_data, y = batch
        seq1, seq2, nano1, nano2, label, label1, label2 = test_batch

        repre1 = self.forward(seq1, nano1)
        logits1 = self.get_logits(seq1, nano1)
        repre2 = self.forward(seq2, nano2)
        logits2 = self.get_logits(seq2, nano2)

        contrast_loss = contrast_cri(repre1, repre2, label)
        ce_loss1 = criterion(logits1, label1)
        ce_loss2 = criterion(logits2, label2)
        loss = contrast_loss + ce_loss1 + ce_loss2

        # calculate the accuracy
        acc_sum1 = (logits1.argmax(dim=1) == label1).float().sum().item()
        acc_sum2 = (logits2.argmax(dim=1) == label2).float().sum().item()
        n = label1.shape[0] + label2.shape[0]
        acc = (acc_sum1 + acc_sum2) / n

        # loss.backward(retain_graph=True)
        output = torch.cat((logits1, logits2), dim=0)
        y = torch.cat((label1, label2), dim=0)
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

        acc = torch.tensor(acc, dtype=float).cuda()

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_ACC", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_AUPRC", AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_AUROC", AUROC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_ACC': acc, 'test_AUPRC': AUPRC, 'test_AUROC': AUROC, 'test_Precision': Precision, 'test_Recall': Recall, 'test_F1Score': F1Score,}

    def test_epoch_end(self, outputs):
        # 计算并输出平均损失
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        avg_ACC = torch.stack([x['test_ACC'] for x in outputs]).mean()
        self.log('avg_test_ACC', avg_ACC, on_step=False, on_epoch=True, prog_bar=True)
        avg_AUPRC = torch.stack([x['test_AUPRC'] for x in outputs]).mean()
        self.log('avg_test_AUPRC', avg_AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        avg_AUROC = torch.stack([x['test_AUROC'] for x in outputs]).mean()
        self.log('avg_test_AUROC', avg_AUROC, on_step=False, on_epoch=True, prog_bar=True)
        avg_Precision = torch.stack([x['test_Precision'] for x in outputs]).mean()
        self.log('avg_test_Precision', avg_Precision, on_step=False, on_epoch=True, prog_bar=True)
        avg_Recall = torch.stack([x['test_Recall'] for x in outputs]).mean()
        self.log('avg_test_Recall', avg_Recall, on_step=False, on_epoch=True, prog_bar=True)
        avg_F1Score = torch.stack([x['test_F1Score'] for x in outputs]).mean()
        self.log('avg_test_F1Score', avg_F1Score, on_step=False, on_epoch=True, prog_bar=True)
        print('avg_test_loss:', avg_loss, '\tavg_test_ACC:', avg_ACC, '\tavg_test_AUPRC:', avg_AUPRC, '\tavg_test_AUROC:', avg_AUROC, '\tavg_test_Precision:', avg_Precision, '\tavg_test_Recall:', avg_Recall, '\tavg_test_F1:', avg_F1Score)

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
