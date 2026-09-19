"""CRAFT-compatible conditional DDPM/DDIM for normalized traffic sequences.

The schedule and epsilon-prediction equations follow the original CRAFT
implementation.  Unlike that implementation, all diffusion coefficients are
registered buffers and clipping is optional because this project uses
``log1p_zscore`` rather than a fixed ``[-1, 1]`` data range.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError("timesteps 必须为正")
    scale = 1000.0 / timesteps
    return torch.linspace(
        scale * 0.0001, scale * 0.02, timesteps, dtype=torch.float64
    )


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError("timesteps 必须为正")
    x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    cumulative = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    return (1.0 - cumulative[1:] / cumulative[:-1]).clamp(0.0, 0.999)


def extract(buffer: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if timesteps.ndim != 1 or timesteps.dtype != torch.long:
        raise ValueError("timesteps 必须是 LongTensor[B]")
    values = buffer.gather(0, timesteps)
    return values.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


@dataclass(frozen=True)
class DiffusionPrediction:
    pred_noise: torch.Tensor
    pred_x0: torch.Tensor


class ConditionalGaussianDiffusion1D(nn.Module):
    """Epsilon-prediction diffusion whose batch axis may be ``B*N`` nodes."""

    def __init__(
        self,
        estimator: nn.Module,
        *,
        data_channels: int,
        seq_length: int,
        time_steps: int = 500,
        sampling_time_steps: int | None = None,
        beta_schedule: str = "linear",
        ddim_sampling_eta: float = 0.0,
        use_self_cond: bool = True,
        clip_x0: bool = False,
        self_condition_probability: float = 0.5,
    ):
        super().__init__()
        sampling_time_steps = time_steps if sampling_time_steps is None else sampling_time_steps
        if min(data_channels, seq_length, time_steps, sampling_time_steps) <= 0:
            raise ValueError("Diffusion channels/length/steps 必须为正")
        if sampling_time_steps > time_steps:
            raise ValueError("sampling_time_steps 不能大于 time_steps")
        if not 0.0 <= self_condition_probability <= 1.0:
            raise ValueError("self_condition_probability 必须在 [0,1]")
        self.estimator = estimator
        self.data_channels = int(data_channels)
        self.seq_length = int(seq_length)
        self.time_steps = int(time_steps)
        self.sampling_time_steps = int(sampling_time_steps)
        self.beta_schedule = str(beta_schedule)
        self.ddim_sampling_eta = float(ddim_sampling_eta)
        self.use_self_cond = bool(use_self_cond)
        self.clip_x0 = bool(clip_x0)
        self.self_condition_probability = float(self_condition_probability)

        if beta_schedule == "linear":
            betas = linear_beta_schedule(time_steps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(time_steps)
        else:
            raise ValueError(f"未知 beta schedule: {beta_schedule}")
        betas = betas.float()
        if not torch.isfinite(betas).all() or (betas <= 0).any() or (betas >= 1).any():
            raise ValueError(
                "beta schedule 产生了不在 (0,1) 的系数；CRAFT linear schedule "
                "需要足够大的 time_steps，tiny 测试请使用 cosine"
            )
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        cumulative_previous = F.pad(cumulative[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - cumulative_previous) / (1.0 - cumulative)
        buffers = {
            "betas": betas,
            "alphas_cumprod": cumulative,
            "alphas_cumprod_prev": cumulative_previous,
            "sqrt_alphas_cumprod": cumulative.sqrt(),
            "sqrt_one_minus_alphas_cumprod": (1.0 - cumulative).sqrt(),
            "log_one_minus_alphas_cumprod": (1.0 - cumulative).log(),
            "sqrt_recip_alphas_cumprod": (1.0 / cumulative).sqrt(),
            "sqrt_recipm1_alphas_cumprod": (1.0 / cumulative - 1.0).sqrt(),
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": posterior_variance.clamp(min=1e-20).log(),
            "posterior_mean_coef1": betas * cumulative_previous.sqrt() / (1.0 - cumulative),
            "posterior_mean_coef2": (1.0 - cumulative_previous) * alphas.sqrt() / (1.0 - cumulative),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value)

    @property
    def sampling_method(self) -> str:
        return "ddim" if self.sampling_time_steps < self.time_steps else "ddpm"

    def _validate_data(self, value: torch.Tensor, condition: torch.Tensor) -> None:
        if value.ndim != 3 or value.shape[1:] != (self.data_channels, self.seq_length):
            raise ValueError(
                f"Diffusion 期望 [B,{self.data_channels},{self.seq_length}]，实得 {tuple(value.shape)}"
            )
        if condition.ndim != 2 or condition.shape[0] != value.shape[0]:
            raise ValueError("Diffusion condition 必须是 [B,D]")
        if not torch.isfinite(value).all() or not torch.isfinite(condition).all():
            raise ValueError("严格模式: Diffusion 输入含 NaN/Inf")

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape or not torch.isfinite(noise).all():
            raise ValueError("q_sample noise shape 非法或含 NaN/Inf")
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_start_from_noise(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def predict_noise_from_start(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, x_start: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t - x_start
        ) / extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)

    def model_predictions(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
    ) -> DiffusionPrediction:
        pred_noise = self.estimator(x_t, timesteps, condition, x_self_cond)
        if pred_noise.shape != x_t.shape or not torch.isfinite(pred_noise).all():
            raise ValueError("严格模式: epsilon estimator 输出非法")
        pred_x0 = self.predict_start_from_noise(x_t, timesteps, pred_noise)
        if self.clip_x0:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
            pred_noise = self.predict_noise_from_start(x_t, timesteps, pred_x0)
        return DiffusionPrediction(pred_noise, pred_x0)

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = (
            extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        variance = extract(self.posterior_variance, timesteps, x_t.shape)
        log_variance = extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return mean, variance, log_variance

    @staticmethod
    def masked_epsilon_loss(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("epsilon prediction/target shape 不一致")
        if mask is None:
            mask = torch.ones_like(target)
        else:
            try:
                mask = torch.broadcast_to(mask.to(prediction.device, prediction.dtype), prediction.shape)
            except RuntimeError as error:
                raise ValueError("Diffusion mask 无法广播到 target") from error
        if not torch.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("Diffusion mask 必须为有限非负值")
        denominator = mask.sum().clamp_min(1.0)
        return ((prediction - target).square() * mask).sum() / denominator

    def training_loss(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        force_self_condition: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_data(x_start, condition)
        batch = x_start.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.time_steps, (batch,), device=x_start.device)
        else:
            timesteps = timesteps.to(device=x_start.device, dtype=torch.long)
        if timesteps.shape != (batch,) or (timesteps < 0).any() or (timesteps >= self.time_steps).any():
            raise ValueError("training timesteps 非法")
        if noise is None:
            noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise)
        use_self = (
            self.use_self_cond
            and (force_self_condition if force_self_condition is not None
                 else random.random() < self.self_condition_probability)
        )
        self_condition = None
        if use_self:
            with torch.no_grad():
                self_condition = self.model_predictions(
                    noisy, timesteps, condition
                ).pred_x0.detach()
        predicted = self.estimator(noisy, timesteps, condition, self_condition)
        if predicted.shape != noise.shape or not torch.isfinite(predicted).all():
            raise ValueError("严格模式: epsilon estimator 输出非法")
        loss = self.masked_epsilon_loss(predicted, noise, mask)
        return loss, {
            "pred_noise": predicted,
            "target_noise": noise,
            "timesteps": timesteps,
            "x_t": noisy,
        }

    @torch.no_grad()
    def _ddpm_sample(
        self, condition: torch.Tensor, initial_noise: torch.Tensor
    ) -> torch.Tensor:
        value = initial_noise
        x_start = None
        for step in reversed(range(self.time_steps)):
            timesteps = torch.full(
                (value.shape[0],), step, device=value.device, dtype=torch.long
            )
            prediction = self.model_predictions(
                value,
                timesteps,
                condition,
                x_start if self.use_self_cond else None,
            )
            mean, _, log_variance = self.q_posterior(
                prediction.pred_x0, value, timesteps
            )
            noise = torch.randn_like(value) if step > 0 else torch.zeros_like(value)
            value = mean + (0.5 * log_variance).exp() * noise
            x_start = prediction.pred_x0
        return value

    @torch.no_grad()
    def _ddim_sample(
        self, condition: torch.Tensor, initial_noise: torch.Tensor
    ) -> torch.Tensor:
        times = torch.linspace(
            -1, self.time_steps - 1, steps=self.sampling_time_steps + 1
        ).int().tolist()
        pairs = list(zip(reversed(times[1:]), reversed(times[:-1])))
        value = initial_noise
        x_start = None
        for step, next_step in pairs:
            timesteps = torch.full(
                (value.shape[0],), step, device=value.device, dtype=torch.long
            )
            prediction = self.model_predictions(
                value,
                timesteps,
                condition,
                x_start if self.use_self_cond else None,
            )
            x_start = prediction.pred_x0
            if next_step < 0:
                value = x_start
                continue
            alpha = self.alphas_cumprod[step]
            alpha_next = self.alphas_cumprod[next_step]
            sigma = self.ddim_sampling_eta * (
                (1.0 - alpha / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha)
            ).sqrt()
            coefficient = (1.0 - alpha_next - sigma.square()).clamp_min(0.0).sqrt()
            noise = torch.randn_like(value) if float(sigma) > 0 else torch.zeros_like(value)
            value = x_start * alpha_next.sqrt() + coefficient * prediction.pred_noise + sigma * noise
        return value

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, int | str | float]]:
        if condition.ndim != 2 or not torch.isfinite(condition).all():
            raise ValueError("Diffusion sampling condition 必须为有限 [B,D]")
        shape = (condition.shape[0], self.data_channels, self.seq_length)
        if initial_noise is None:
            initial_noise = torch.randn(shape, device=condition.device, dtype=condition.dtype)
        if initial_noise.shape != shape or not torch.isfinite(initial_noise).all():
            raise ValueError("Diffusion initial_noise shape 非法或含 NaN/Inf")
        if self.sampling_method == "ddpm":
            result = self._ddpm_sample(condition, initial_noise)
            steps = self.time_steps
        else:
            result = self._ddim_sample(condition, initial_noise)
            steps = self.sampling_time_steps
        return result, {
            "method": self.sampling_method,
            "sampling_steps": steps,
            "training_steps": self.time_steps,
            "ddim_eta": self.ddim_sampling_eta,
        }


# Name retained for readers comparing this module with original CRAFT.
GaussianDiffusion1D = ConditionalGaussianDiffusion1D
