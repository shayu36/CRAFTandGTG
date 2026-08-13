import ast
import json
import random
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from os.path import join

from tqdm import tqdm

from data_loaders import get_test_dataloader
from craft import CRAFTModel


def generate_data(model, test_loader):
    """
    返回归一化的生成数据
    """
    model.eval()
    x_gen = []
    region_ids = []
    start_hours, dates, weekdays, months = [], [], [], []
    for batch in tqdm(test_loader, total=len(test_loader), desc='test'):
        with torch.no_grad():
            x = model.generate(batch)
        # 注意维度 squeeze
        x_gen.append(x.squeeze().cpu().numpy())
        region_ids += batch['region_id']
        dates += batch['date']
        start_hours += batch['start_hour'].detach().cpu().tolist()
        weekdays += batch['weekday'].detach().cpu().tolist()
        months += batch['month'].detach().cpu().tolist()
    x_gen = np.concatenate(x_gen)
    # 将生成结果映射到 [0, 1]
    x_gen = np.clip(x_gen, a_min=-1, a_max=1)
    x_gen = (x_gen + 1) / 2

    gen_data_list = [
        {
            'region_id': region_ids[i],
            'start_hour': start_hours[i],
            'date': dates[i],
            'weekday': weekdays[i],
            'month': months[i],
            'is_weekend': weekdays[i] > 4,
            'in_flow': x_gen[i][0].tolist(),
            'out_flow': x_gen[i][1].tolist()
        }
        for i in range(len(region_ids))
    ]
    gen_df = pd.DataFrame(gen_data_list)
    return gen_df


def run_generation(cfg):
    exp_dir = cfg['exp_dir']

    loaders_dict = get_test_dataloader(cfg)

    model = CRAFTModel(cfg).to(cfg['device'])
    state_dict = torch.load(join(exp_dir, f'craft.pth'), map_location=cfg['device'], weights_only=True)
    model.load_state_dict(state_dict=state_dict['ema_model'])

    for city, test_loader in loaders_dict.items():
        gen_df = generate_data(model, test_loader)
        gen_df.to_csv(join(exp_dir, f'craft_gen_{city}.csv'), index=False)
