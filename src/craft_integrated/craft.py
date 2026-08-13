import torch
import torch.nn as nn

from diffusion import GaussianDiffusion1D


class CRAFTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg['device']
        self.feature_dim = cfg['rep_dim']
        self.data_channels = cfg['data_channels']
        self.cond_emb_dim = cfg['cond_emb_dim']
        self.use_retrieve = cfg['use_retrieve']
        self.refer_dim = cfg['refer_dim']
        self.use_month = cfg['use_month']

        self.hour_num = 24
        self.hour_dim = cfg['hour_dim']
        self.weekday_num = 7
        self.weekday_dim = cfg['weekday_dim']
        self.month_num = 12
        self.month_dim = cfg['month_dim']

        self.hour_embedding = nn.Embedding(self.hour_num, self.hour_dim)
        self.weekday_embedding = nn.Embedding(self.weekday_num, self.weekday_dim)
        self.input_cond_dim = self.feature_dim + self.hour_dim + self.weekday_dim + self.refer_dim

        if self.use_month:
            self.month_embedding = nn.Embedding(self.month_num, self.month_dim)
            self.input_cond_dim += self.month_dim

        # 提取时间特征
        self.temporal_encoder = ReferTransformer(cfg)

        self.cond_mlp = nn.Sequential(
            nn.Linear(self.input_cond_dim, self.cond_emb_dim),
            nn.ReLU(),
            nn.Linear(self.cond_emb_dim, self.cond_emb_dim)
        )
        self.generator_model = GaussianDiffusion1D(cfg)

    def get_cond(self, batch):
        feature = batch['feature'].to(self.device)
        start_hour = batch['start_hour'].to(self.device)
        weekday = batch['weekday'].to(self.device)
        hour_emb = self.hour_embedding(start_hour)
        weekday_emb = self.weekday_embedding(weekday)
        cond_list = [feature, hour_emb, weekday_emb]

        if self.use_month:
            month = batch['month'].to(self.device)
            month_emb = self.month_embedding(month)
            cond_list.append(month_emb)

        reference = batch['reference'].to(self.device)
        reference = self.temporal_encoder(reference)
        cond_list.append(reference)

        cond = torch.cat(cond_list, dim=-1)
        cond = self.cond_mlp(cond)
        return cond

    def calc_loss(self, batch):
        cond = self.get_cond(batch)
        loss = self.generator_model.calc_loss({'x': batch['x'], 'cond': cond})
        return loss

    def generate(self, batch):
        cond = self.get_cond(batch)
        return self.generator_model.generate(batch_size=cond.shape[0], cond=cond)


def positional_encoding(max_seq_length, d_model):
    pe = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ReferTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg['device']
        self.data_channels = cfg['data_channels']
        self.seq_length = cfg['seq_length']
        self.d_model = cfg['refer_dim']
        self.head_num = cfg['refer_head_num']
        self.refer_layer_num = cfg['refer_layer_num']
        self.pos_encoder = nn.Embedding.from_pretrained(positional_encoding(self.seq_length, self.d_model), freeze=True)

        self.input_proj = nn.Linear(self.data_channels, self.d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.head_num, dim_feedforward=self.d_model, batch_first=True
            ),
            self.refer_layer_num
        )

    def forward(self, x):
        """
        x: (bs, 2, seq_length)
        """
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.transformer(x)
        return torch.mean(x, dim=1)
