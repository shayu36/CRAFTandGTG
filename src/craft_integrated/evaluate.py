import ast
import numpy as np
import pandas as pd
from os.path import join

from data_loaders import load_norm_flow



def calc_cpc(x_pred, x_true):
    """
    Common part of commuters (Sørensen-Dice index)
    """
    return np.sum(2 * np.minimum(x_pred, x_true)) / (np.sum(x_pred) + np.sum(x_true))


def calc_norm_rmse(x_pred, x_true):
    return np.sqrt(np.mean((x_pred - x_true) ** 2)) / (x_true.max() - x_true.min())


def calc_norm_mae(x_pred, x_true):
    return np.mean(np.abs(x_pred - x_true)) / (x_true.max() - x_true.min())


def calc_group_evaluate_metric(real_df: pd.DataFrame, gen_df: pd.DataFrame):
    real_df = real_df.sort_values(by=['date', 'start_hour']).reset_index(drop=True)
    gen_df = gen_df.sort_values(by=['date', 'start_hour']).reset_index(drop=True)
    assert len(real_df) == len(gen_df)
    assert np.sum(real_df['date'].to_numpy() == gen_df['date'].to_numpy()) == len(real_df)
    assert np.sum(real_df['start_hour'].to_numpy() == gen_df['start_hour'].to_numpy()) == len(real_df)
    real_df = (real_df.groupby(by=['region_id', 'month', 'weekday', 'start_hour'])[['in_flow', 'out_flow']].mean()
               .reset_index().sort_values(by=['region_id', 'month', 'weekday', 'start_hour']))
    gen_df = (gen_df.groupby(by=['region_id', 'month', 'weekday', 'start_hour'])[['in_flow', 'out_flow']].mean()
               .reset_index().sort_values(by=['region_id', 'month', 'weekday', 'start_hour']))

    eval_list = []
    for tp in ['in_flow', 'out_flow']:
        real_values = np.stack(real_df[tp].tolist(), axis=0)
        gen_values = np.stack(gen_df[tp].tolist(), axis=0)
        eval_list.append({
            'data_type': tp,
            'cpc': calc_cpc(gen_values, real_values),
            'min_max_mae': calc_norm_mae(gen_values, real_values),
            'min_max_rmse': calc_norm_rmse(gen_values, real_values)
        })
    result = pd.DataFrame(eval_list)
    return result


def load_gen_data(exp_dir, method, city):
    gen_df = pd.read_csv(join(exp_dir, city, f'{method}_gen_{city}.csv'))
    gen_df['in_flow'] = gen_df['in_flow'].apply(lambda x: np.array(ast.literal_eval(x)))
    gen_df['out_flow'] = gen_df['out_flow'].apply(lambda x: np.array(ast.literal_eval(x)))
    gen_df['is_weekend'] = gen_df['weekday'].apply(lambda x: x > 4)
    gen_df['month'] = gen_df['date'].apply(lambda ts: pd.Timestamp(ts).month - 1)
    gen_df['date'] = gen_df['date'].apply(lambda ts: pd.Timestamp(ts).date())
    return gen_df


def run_evaluate(exp_dir, method, seq_length=24, cities=None):
    print(f'Evaluate {exp_dir} {method} ...... ')
    if cities is None:
        cities = ['chi', 'dc', 'toronto', 'ny']
    eval_dfs = []
    for city in cities:
        real_df = load_norm_flow(city=city, seq_length=seq_length, phase='test')
        real_df['in_flow'] = real_df['norm_flow'].apply(lambda x: x[0])
        real_df['out_flow'] = real_df['norm_flow'].apply(lambda x: x[1])

        gen_df = load_gen_data(exp_dir=exp_dir, method=method, city=city)
        eval_df = calc_group_evaluate_metric(real_df=real_df, gen_df=gen_df)

        eval_df['city'] = city
        eval_dfs.append(eval_df)
    eval_df = pd.concat(eval_dfs)

    save_file = f'{method}_eval.csv'
    eval_df.to_csv(join(exp_dir, save_file), index=False)
    print(eval_df.to_string())
    return eval_df


if __name__ == '__main__':
    seq_len = 24
    run_evaluate(exp_dir=f'ckpt/exp_0/seq_{seq_len}', method='craft', seq_length=seq_len)
