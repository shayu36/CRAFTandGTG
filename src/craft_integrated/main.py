import os
import yaml
from os.path import join
from argparse import ArgumentParser
import data_loaders
from train import pretrain_rep_model, train_diffusion
from generate import run_generation
from utils import set_random_time_seed


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--seq_length', type=int, default=24)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--src_cities', nargs='+', type=str)
    parser.add_argument('--trg_cities', nargs='+', type=str)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    config.update({k: v for k, v in vars(args).items() if v is not None})

    config['seed'] = set_random_time_seed()

    # for debug (仅当配置未显式指定时使用小 epoch)
    config.setdefault('pretrain_epoch', 20)
    config.setdefault('diff_train_epoch', 3)

    # 注入数据路径与 GTG 融合开关 (只读 CRAFT, 归一化/缓存写在 Paper)
    data_loaders.configure(config)

    print(config)
    trg_dir = f'seq_{args.seq_length}/' + '_'.join(config['trg_cities'])
    config['exp_dir'] = f'ckpt/exp_{config["exp_id"]}/seq_{config["seq_length"]}/{trg_dir}'

    os.makedirs(config['exp_dir'], exist_ok=True)
    yaml.dump(config, open(join(config['exp_dir'], 'train_config.yaml'), 'w'), default_flow_style=False)

    # 训练表征模型, 做表征对齐
    pretrain_rep_model(config)

    # 训练扩散模型
    train_diffusion(config)

    # 生成目标城市的数据
    run_generation(config)

