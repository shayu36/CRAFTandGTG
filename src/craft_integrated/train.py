import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from os.path import join
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ema_pytorch import EMA
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler

from rep_model import GTAggregator
from craft import CRAFTModel
from data_loaders import FlowDataset, get_graph_datasets, get_source_train_datasets


class EarlyStop:
    def __init__(self, patience):
        self.patience = patience
        self.min_loss = float('inf')
        self.counter = 0

    def update(self, loss):
        if loss < self.min_loss:
            self.counter = 0
            self.min_loss = loss
        else:
            self.counter += 1
        return self.counter > self.patience


def pretrain_rep_model(cfg):
    src_cities, trg_cities = cfg['src_cities'], cfg['trg_cities']
    exp_dir = cfg['exp_dir']
    os.makedirs(exp_dir, exist_ok=True)

    model = GTAggregator(cfg).to(cfg['device'])
    graph_dict = get_graph_datasets(src_cities=src_cities, trg_cities=trg_cities)

    optimizer = Adam(params=model.parameters(), lr=cfg['lr'])

    min_loss = math.inf
    pretrain_rcd = []
    for epoch in range(cfg['pretrain_epoch']):
        loss, loss_items = model.calc_loss(graph_dict)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Pretrain epoch {epoch} total loss {loss.item():.6f} '
              f'sim_loss {loss_items["sim_loss"]:.6f} '
              f'w_loss {loss_items["w_loss"]:.6f}')

        epoch_loss = loss.item()
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), join(exp_dir, 'rep_model.pth'))
        pretrain_rcd.append({'epoch': epoch, 'loss': epoch_loss})
        pd.DataFrame(pretrain_rcd).to_csv(join(exp_dir, 'pretrain_loss.csv'), index=False)

    model.load_state_dict(torch.load(join(exp_dir, 'rep_model.pth'), map_location=cfg['device'], weights_only=True))
    model.eval()
    for graph in graph_dict['src_graphs'] + graph_dict['trg_graphs']:
        city = graph.city
        with torch.no_grad():
            reps = model(graph.x, graph.edge_index)
        reps = reps.detach().cpu().numpy()
        np.save(join(exp_dir, f'{city}_rep.npy'), reps)


def build_balanced_sample_dataloader(datasets, samples_per_dataset, batch_size):
    offsets = [0]
    for dataset in datasets[:-1]:
        offsets.append(offsets[-1] + len(dataset))
    all_sampled_indices = []
    for i, dataset in enumerate(datasets):
        dataset_size = len(dataset)
        if dataset_size < samples_per_dataset:
            indices = np.random.choice(dataset_size, samples_per_dataset, replace=True)
        else:
            indices = np.random.choice(dataset_size, samples_per_dataset, replace=False)
        all_sampled_indices.extend(indices + offsets[i])
    np.random.shuffle(all_sampled_indices)
    combined_dataset = ConcatDataset(datasets)
    sampler = SubsetRandomSampler(all_sampled_indices)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler, collate_fn=FlowDataset.collator)
    return dataloader


def train_diffusion(cfg):
    exp_dir = cfg['exp_dir']
    batch_size = cfg['batch_size']
    sample_num_dict = cfg['sample_num']

    model = CRAFTModel(cfg).to(cfg['device'])

    optimizer = AdamW(params=model.parameters(), lr=cfg['lr'])
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, min_lr=0.05 * cfg['lr'])
    ema = EMA(model, beta=cfg['ema_decay'], update_every=cfg['update_every'])
    train_dataset_list = get_source_train_datasets(cfg, phase='train')
    valid_dataset_list = get_source_train_datasets(cfg, phase='test')

    def train_epoch():
        model.train()
        mean_loss = 0
        train_loader = build_balanced_sample_dataloader(
            datasets=train_dataset_list, samples_per_dataset=sample_num_dict['train'], batch_size=batch_size
        )
        for itr, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='train epoch'):
            optimizer.zero_grad()
            loss = model.calc_loss(batch)
            loss.backward()
            optimizer.step()
            ema.update()
            mean_loss += loss.item()
        mean_loss /= len(train_loader)
        return mean_loss

    def valid_epoch():
        valid_model = ema.ema_model
        valid_model.eval()
        valid_loader = build_balanced_sample_dataloader(
            datasets=valid_dataset_list, samples_per_dataset=sample_num_dict['test'], batch_size=batch_size
        )
        mean_loss = 0
        for itr, batch in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='valid epoch'):
            with torch.no_grad():
                loss = valid_model.calc_loss(batch)
            mean_loss += loss.item()
        valid_model.train()  #
        mean_loss /= len(valid_loader)
        return mean_loss

    stop_checker = EarlyStop(patience=30)
    min_valid_loss = math.inf
    diff_train_rcd = []
    for epoch in range(cfg['diff_train_epoch']):
        train_loss = train_epoch()
        valid_loss = valid_epoch()
        print(f'Epoch {epoch} train loss {train_loss:.4f} valid loss {valid_loss:.4f}')
        # 记录损失
        diff_train_rcd.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss
        })
        pd.DataFrame(data=diff_train_rcd).to_csv(join(exp_dir, f'craft_train_loss.csv'), index=False)

        lr_scheduler.step(valid_loss)
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save({
                'ori_model': model.state_dict(),
                'ema_model': ema.ema_model.state_dict()
            }, join(exp_dir, f'craft.pth'))
        early_stop = stop_checker.update(valid_loss)
        if epoch > 200 and early_stop:
            print(f'Early stop at epoch {epoch}')
            break
