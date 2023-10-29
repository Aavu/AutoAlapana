import os
import torch
import math
from tqdm import tqdm
import numpy as np
from alaapNet.data.dataset import AlapanaDataset
from torch.utils.data import DataLoader
from alaapNet.models.alaapNet import Seq2Seq, Seq2SeqComponent, TransformerModel
from alaapNet.models.modules import AlaapLoss
from alaapNet.models.conv_seq2seq import ConvSeq2Seq
from alaapNet.models.gamakanet import GamakaNet
from alaapNet.models.rnn import GRUTimeSeries
import matplotlib.pyplot as plt
from alaapNet.tools.utils import Util


class AlapanaTrainer:
    def __init__(self, dataset_path, in_seq_len, out_seq_len, hop_length, hidden_size, num_layers,
                 batch_size, lr, resample_factor=1, max_value=3.0):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'mps' if torch.backends.mps.is_available else 'cpu')
        # self.device = 'cpu'
        # assert (input_size & (input_size - 1) == 0) and input_size != 0  # Make sure input is power of 2
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr

        self.train_data = AlapanaDataset(dataset_path,
                                         train=True,
                                         input_length=self.in_seq_len,
                                         target_length=self.out_seq_len,
                                         hop_length=hop_length,
                                         resample_factor=resample_factor,
                                         max_value=max_value)
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.val_data = AlapanaDataset(dataset_path,
                                       train=False,
                                       input_length=self.in_seq_len,
                                       target_length=self.out_seq_len,
                                       hop_length=hop_length,
                                       resample_factor=resample_factor,
                                       max_value=max_value)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)

        # self.model = GRUTimeSeries(hidden_size=self.hidden_size,
        #                            num_recurrent_layers=self.num_layers,
        #                            num_fc_layers=self.num_layers,
        #                            out_seq_len=self.out_seq_len).to(self.device)

        self.model = Seq2Seq(feature_size=1,
                             hidden_size=self.hidden_size,
                             target_indices=[0],
                             num_layers=self.num_layers,
                             dropout_p=0).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.L1Loss()  # torch.nn.MSELoss() # AlaapLoss() # torch.nn.CrossEntropyLoss() -> for bytecode prediction

        self.train_losses = []
        self.val_losses = []
        self.best_train_loss = np.inf
        self.ckpt_path = None

    # inverse sigmoid decay from https://arxiv.org/pdf/1506.03099.pdf
    @staticmethod
    def compute_teacher_force_prob(idx, decay):
        return decay / (decay + math.exp(idx / decay))

    def train(self, epochs=10, ckpt_dir="checkpoints", restore_ckpt_path=None):
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        self.ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")

        if restore_ckpt_path is not None:
            ckpt = torch.load(restore_ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optim_state'])

        self.model.train()

        for epoch in range(epochs):
            train_loss = 0
            tf_prob = self.compute_teacher_force_prob(epoch, decay=10)
            for X, y in tqdm(self.train_loader, total=len(self.train_loader)):
                X = X.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X, y, 0)
                if torch.any(torch.isnan(pred)):
                    print("prediction nan")
                    exit()
                loss = self.criterion(pred, y)
                loss.backward(retain_graph=True)
                train_loss += loss.item()
                self.optimizer.step()

            train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(train_loss)

            improved_train = train_loss < self.best_train_loss

            print(f'Epoch {epoch + 1} / {epochs} '
                  f'\t Train Loss = {train_loss:.4f} ')

            if improved_train:
                print(f"\t (Train Loss improved ({self.best_train_loss:.4f} ---> {train_loss:.4f}))")
                self.best_train_loss = train_loss
                checkpoint = {'epoch': epoch,
                              'loss': self.best_train_loss,
                              'model_state': self.model.state_dict(),
                              'optim_state': self.optimizer.state_dict()}
                torch.save(checkpoint, self.ckpt_path)

    def eval(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        val_loss = 0
        for X, y in tqdm(self.val_loader, total=len(self.val_loader)):
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X, y, 0)
            loss = self.criterion(pred, y)
            for b in range(len(X)):
                x = Util.to_numpy(y[b].view(-1))
                p = Util.to_numpy(pred[b].view(-1))
                plt.plot(x)
                plt.plot(p)
                plt.show()

                val_loss += loss.item()
            break
        val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(val_loss)
