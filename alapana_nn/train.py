import os
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torchvision.ops import sigmoid_focal_loss
from alapana_nn.dataset import SilenceDataset, SilenceDataloader
from alapana_nn.models import SilencePredictorModel, SilencePredictorModel2
from alapana_nn.utils import Util


class Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return sigmoid_focal_loss(pred, target, self.alpha, self.gamma, self.reduction) + self.ce_loss(pred, target)


class SilenceTrainer:
    def __init__(self, dataset_path, batch_size, lr, hidden_size, num_layers):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.divisions = 1
        train_data = SilenceDataset(root=dataset_path, train=True, divisions=self.divisions)
        self.max_silence = train_data.max_silence_length
        self.train_loader = SilenceDataloader(train_data, shuffle=True, batch_size=batch_size)
        val_data = SilenceDataset(root=dataset_path, train=False, divisions=self.divisions)
        self.val_loader = SilenceDataloader(val_data, shuffle=False, batch_size=batch_size)

        self.model = SilencePredictorModel2(feature_size=1,
                                            output_size=self.divisions,
                                            hidden_size=self.hidden_size,
                                            num_layers=self.num_layers).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)
        self.criterion = nn.L1Loss()

        self.train_losses = []
        self.val_losses = []
        self.best_train_loss = np.inf
        self.ckpt_path = None

    def train(self, epochs=10, ckpt_dir="checkpoints", restore_ckpt_path=None):
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        self.ckpt_path = os.path.join(ckpt_dir, "checkpoint-silence.pth")

        if restore_ckpt_path is not None:
            ckpt = torch.load(restore_ckpt_path, map_location=self.device)
            self.best_train_loss = ckpt["loss"]
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optim_state'])

        for epoch in range(epochs):
            train_loss = 0
            # acc = 0
            self.model.train()
            for current_phrase, next_phrase, target, lengths in tqdm(self.train_loader, total=len(self.train_loader)):
                current_phrase = current_phrase.to(self.device)
                next_phrase = next_phrase.to(self.device)
                target = target.to(self.device)
                lengths = lengths.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(current_phrase=current_phrase,
                                  next_phrase=next_phrase,
                                  lengths=lengths)
                if torch.any(torch.isnan(pred)):
                    print("prediction nan")
                    exit()

                loss = self.criterion(pred, target)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                # acc += torch.mean((torch.argmax(target, dim=-1) == torch.argmax(pred, dim=-1)).float())

            train_loss = train_loss / len(self.train_loader)
            # acc = acc / len(self.train_loader)
            self.train_losses.append(train_loss)

            improved_train = train_loss < self.best_train_loss

            print(f'Epoch {epoch + 1} / {epochs} '
                  f'\t Train Loss = {train_loss:.4f} '
                  # f'\t Acc = {acc: .4f}'
                  )
            if improved_train:
                print(f"\t (Train Loss improved ({self.best_train_loss:.4f} ---> {train_loss:.4f}))")
                self.best_train_loss = train_loss
                checkpoint = {'epoch': epoch,
                              'loss': self.best_train_loss,
                              'max_silence': self.max_silence,
                              'model_state': self.model.state_dict(),
                              'optim_state': self.optimizer.state_dict()}
                torch.save(checkpoint, self.ckpt_path)

                with torch.no_grad():
                    val_loss = self.eval()
                    print(f"Val Loss: {val_loss:.4f} "
                          # f"\t Val Accuracy: {acc: .4f}"
                          )

    def eval(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])

        self.model.eval()
        loader = self.val_loader
        val_loss = 0
        # acc = 0
        for x1, x2, target, lengths in tqdm(loader, total=len(loader)):
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            target = target.to(self.device)
            lengths = lengths.to(self.device)
            pred = self.model(current_phrase=x1, next_phrase=x2,
                              lengths=lengths)
            loss = self.criterion(pred, target)

            val_loss += loss.item()
            # acc += torch.mean((torch.argmax(target, dim=-1) == torch.argmax(pred, dim=-1)).float())

        # acc = acc / len(loader)
        val_loss = val_loss / len(loader)
        self.val_losses.append(val_loss)
        return val_loss
