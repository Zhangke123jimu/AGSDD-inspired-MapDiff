import copy
import datetime
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from pathlib import Path
from tqdm import tqdm


class Trainer(object):
    def __init__(self, config, model, optimizer, epochs, loss_fn, train_dataloader, output_dir, device,
                 scheduler=None, experiment=None, save_epochs=10, resume_path=None,distributed=False,is_main_process=True,train_sampler=None,):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment = experiment
        self.train_dataloader = train_dataloader
        self.loss_fn = loss_fn
        self.train_num_epochs = epochs
        self.step = 0
        self.device = device
        self.output_dir = output_dir
        self.save_epochs = save_epochs
        self.distributed = distributed
        self.is_main_process = is_main_process
        self.train_sampler = train_sampler
        Path(self.output_dir + '/model/').mkdir(exist_ok=True)

        if resume_path is None:
            self.global_epoch = 0
        else:
            self.global_epoch = torch.load(resume_path,map_location=device)['global_epoch']

        self.current_epoch = 0 + self.global_epoch

    def train_epoch(self):
        self.model.train()
        self.current_epoch += 1
        all_logits = []
        all_labels = []
        all_index = []
        for idx, data in enumerate(tqdm(self.train_dataloader, disable=not self.is_main_process)):
            x, x_pos, x_pad, x_mask, aa_label = data[0].to(self.device), data[1].to(self.device), \
                data[2].to(self.device), data[3].to(self.device), data[4].to(self.device)
            # print(x.shape)
            logits = self.model(x, x_pos, x_mask, x_pad)
            loss = self.loss_fn(logits[x_mask >= 1], aa_label[x_mask >= 1]).mean()
            loss.backward()
            # print(loss.item())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            self.step += 1

            all_logits.append(logits.view(-1, 20).detach().cpu())
            all_labels.append(aa_label.view(-1).detach().cpu())
            all_index.append(x_mask.view(-1).detach().cpu())
            # print("Epoch: %d, batch_idx: %d, Loss: %f" % (self.current_epoch, idx, loss.item()))
            if self.experiment and idx % 10 == 0:
                self.experiment.log_metric('train_loss', loss.item(), step=self.step, epoch=self.current_epoch)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_index = torch.cat(all_index, dim=0)

        if self.is_main_process:
            print("Training metrics:")
        return self.log_metrics(all_labels, all_logits, all_index)

    def save_model(self, save_epochs, curr_loss, mode='best'):
        def unwrap_model(model):
            return model.module if hasattr(model, "module") else model

        config_dict = OmegaConf.to_container(self.config, resolve=True)
        if mode == 'last':
            data = {
                'config': config_dict,
                'epoch': save_epochs,
                'model': unwrap_model(self.model).state_dict(),
                'opt': self.optimizer.state_dict(),
                'scheduler':self.scheduler.state_dict() if self.scheduler else None,
                'global_epoch': self.current_epoch,
                'loss': curr_loss
            }
        elif mode == 'curr':
            data = {
                'config': config_dict,
                'epoch': save_epochs,
                'model': unwrap_model(self.model).state_dict(),
                'opt': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'global_epoch': self.current_epoch,
                'loss': curr_loss
            }
        else:
            raise ValueError(f"unknown mode {mode}")
        save_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(data, os.path.join(self.output_dir, 'model',
                                      f'{self.config.experiment.name}_{mode}_{save_epochs}_epochs_{save_time}.pt'))

    def log_metrics(self, sl_labels, sl_predictions, sl_index):
        all_loss = self.loss_fn(sl_predictions[sl_index >= 1], sl_labels[sl_index >= 1]).mean()
        mask_loss = self.loss_fn(sl_predictions[sl_index == 1], sl_labels[sl_index == 1]).mean()
        replace_loss = self.loss_fn(sl_predictions[sl_index == 2], sl_labels[sl_index == 2]).mean()
        keep_loss = self.loss_fn(sl_predictions[sl_index == 3], sl_labels[sl_index == 3]).mean()

        pred_labels = np.argmax(sl_predictions.numpy(), axis=-1)
        labels = sl_labels.numpy()
        index = sl_index.numpy()

        macro_mask_f1 = f1_score(labels[index == 1], pred_labels[index == 1], average='macro')
        macro_replace_f1 = f1_score(labels[index == 2], pred_labels[index == 2], average='macro')
        macro_keep_f1 = f1_score(labels[index == 3], pred_labels[index == 3], average='macro')

        micro_mask_f1 = f1_score(labels[index == 1], pred_labels[index == 1], average='micro')
        micro_replace_f1 = f1_score(labels[index == 2], pred_labels[index == 2], average='micro')
        micro_keep_f1 = f1_score(labels[index == 3], pred_labels[index == 3], average='micro')

        if self.is_main_process:
            print("Epoch: %d, all_loss: %f, mask_loss: %f, replace_loss: %f, keep_loss: %f" % (
                self.current_epoch, all_loss.item(), mask_loss.item(), replace_loss.item(), keep_loss.item()))
            print("Macro Mask F1: %f, Macro Replace F1: %f, Macro Keep F1: %f" % (
                macro_mask_f1, macro_replace_f1, macro_keep_f1))
            print("Micro Mask F1: %f, Micro Replace F1: %f, Micro Keep F1: %f" % (
                micro_mask_f1, micro_replace_f1, micro_keep_f1))

        return all_loss, macro_mask_f1, macro_replace_f1, macro_keep_f1, micro_mask_f1, micro_replace_f1, micro_keep_f1

    def fit(self):
        if self.current_epoch >= self.train_num_epochs and self.is_main_process:
            print(f"Already reached target epoch {self.train_num_epochs}, current epoch is {self.current_epoch}.")
            return

        while self.current_epoch < self.train_num_epochs:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.current_epoch)
            train_loss, train_mac_mask_f1, train_mac_replace_f1, train_mac_keep_f1, train_mic_mask_f1, \
            train_mic_replace_f1, train_mic_keep_f1 = self.train_epoch()
            if self.experiment and self.is_main_process:
                self.experiment.log_metric('train_macro_mask_f1', train_mac_mask_f1, epoch=self.current_epoch)
                self.experiment.log_metric('train_macro_replace_f1', train_mac_replace_f1, epoch=self.current_epoch)
                self.experiment.log_metric('train_macro_keep_f1', train_mac_keep_f1, epoch=self.current_epoch)
                self.experiment.log_metric('train_micro_mask_f1', train_mic_mask_f1, epoch=self.current_epoch)
                self.experiment.log_metric('train_micro_replace_f1', train_mic_replace_f1, epoch=self.current_epoch)
                self.experiment.log_metric('train_micro_keep_f1', train_mic_keep_f1, epoch=self.current_epoch)
            if self.current_epoch % self.save_epochs == 0 and self.current_epoch > 10 and self.is_main_process:
                self.save_model(self.current_epoch, train_loss, mode='curr')

            torch.cuda.empty_cache()

        if self.is_main_process:
            self.save_model(self.current_epoch, train_loss, mode='last')
