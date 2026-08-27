import math
import yaml
from pathlib import Path
from random import random
from functools import partial
import torch
from torch.nn import Module
import torch.nn.functional as F
from unet import Unet1D


def load_noise_estimator(cfg):
    def update_cfg(global_cfg, local_cfg):
        for key, val in global_cfg.items():
            if key not in local_cfg:
                continue
            local_cfg[key] = val
        return local_cfg

    estimator_cfg_path = Path(__file__).resolve().parent / 'configs' / 'unet.yaml'
    estimator_cfg = yaml.safe_load(open(estimator_cfg_path, 'r'))
    estimator_cfg = update_cfg(cfg, estimator_cfg)
    model = Unet1D(estimator_cfg).to(cfg['device'])
    return model


# Reference: https://github.com/lucidrains/denoising-diffusion-pytorch
# self condition trick paper: Analog bits: Generating discrete data using diffusion models with self-conditioning
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(Module):
    def __init__(self, cfg):
        super().__init__()
        self.data_channels = cfg['data_channels']
        self.use_self_cond = cfg['use_self_cond']
        self.seq_length = cfg['seq_length']
        self.beta_schedule = cfg['beta_schedule']
        self.time_steps = cfg['time_steps']
        self.sampling_time_steps = cfg['sampling_time_steps']
        self.ddim_sampling_eta = cfg['ddim_sampling_eta']

        self.device = cfg['device']
        self.estimator = load_noise_estimator(cfg)

        if self.beta_schedule == 'linear':
            betas = linear_beta_schedule(self.time_steps).to(torch.float32).to(self.device)
        elif self.beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.time_steps).to(torch.float32).to(self.device)
        else:
            raise ValueError(f'unknown beta schedule {self.beta_schedule}')
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        # sampling related parameters
        # self.sampling_timesteps = default(sampling_timesteps, timesteps)
        # default num sampling timesteps to number of timesteps at training
        assert self.sampling_time_steps <= self.time_steps
        self.is_ddim_sampling = self.sampling_time_steps < self.time_steps

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        # for q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = posterior_variance

        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        """
        根据 x_0 和 x_t 计算 x_{t-1} 分布的均值和方差
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, *, cond=None, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        pred_noise = self.estimator(x, t, cond, x_self_cond)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else (lambda y: y)
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    @torch.no_grad()
    def step_sample(self, x, t: int, cond: torch.Tensor, x_self_cond=None, clip_denoised=True):
        """
        给定 x_t, 采样出 x_{t-1}
        """
        batch_size = x.shape[0]
        batched_times = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        pred_noise, x_start = self.model_predictions(x, batched_times, cond=cond, x_self_cond=x_self_cond)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, variance, log_variance = self.q_posterior(x_start=x_start, x_t=x, t=batched_times)

        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def loop_sample(self, batch_size, cond):
        """
        生成的采样步数等于训练的采样步数
        """
        x = torch.randn((batch_size, self.data_channels, self.seq_length), device=self.device)
        x_start = None
        for t in reversed(range(0, self.time_steps)):
            self_cond = x_start if self.use_self_cond else None
            x, x_start = self.step_sample(x, t, cond, self_cond)
        return x

    @torch.no_grad()
    def skip_sample(self, batch_size, cond: torch.Tensor, clip_denoised=True):
        """
        生成的采样步数小于训练的采样步数
        """
        total_steps, sampling_steps, eta = self.time_steps, self.sampling_time_steps, self.ddim_sampling_eta
        times = torch.linspace(-1, total_steps - 1, steps=sampling_steps + 1)
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn((batch_size, self.data_channels, self.seq_length), device=self.device)

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.use_self_cond else None

            pred_noise, x_start = self.model_predictions(
                x, time_cond, cond=cond, x_self_cond=self_cond, clip_x_start=clip_denoised
            )
            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return x

    @torch.no_grad()
    def generate(self, batch_size, cond=None):
        """
        生成用, 根据 cond 生成数据
        :param batch_size
        :param cond: (batch_size, cond_dim)
        :return:
        """
        # batch_size = cond.shape[0]
        if cond is not None:
            assert batch_size == cond.shape[0]
        if self.is_ddim_sampling:
            return self.skip_sample(batch_size=batch_size, cond=cond, clip_denoised=True)
        else:
            return self.loop_sample(batch_size=batch_size, cond=cond)

    def add_noise(self, x_start, t):
        """
        训练用, 加噪声
        """
        noise = torch.randn_like(x_start)
        return noise, (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def calc_loss(self, batch):
        """
        :param batch: Dict
        :return:
        """
        x_start = batch['x'].to(self.device)
        cond = batch['cond'].to(self.device)
        b, c, n = x_start.shape
        assert n == self.seq_length, f'seq length must be {self.seq_length}'
        t = torch.randint(0, self.time_steps, (b,), device=self.device).long()

        # noise sample
        noise, x = self.add_noise(x_start=x_start, t=t)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.use_self_cond and random() < 0.5:
            with torch.no_grad():
                _, x_self_cond = self.model_predictions(x, t, cond=cond)
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.estimator(x, t, cond, x_self_cond)
        loss = F.mse_loss(model_out, noise)
        return loss
