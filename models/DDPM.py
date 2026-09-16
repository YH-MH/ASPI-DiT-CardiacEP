import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial
from copy import deepcopy
from tqdm import tqdm

# import methods
import sys
from utils.misc import instantiate_from_config, exists, default
from models.models_utils import EMA, extract, make_beta_schedule


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class GaussianDiffusion(pl.LightningModule):
    """Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.
    Input:
        x: tensor of shape (N, input_dim)
        y: tensor of shape (标量)
    Output:
        scalar loss tensor
    Args:
        model (pl.LightningModule): model which estimates diffusion noise
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """
    def __init__(
            self,
            diff_model_config,

            given_betas=None,
            timesteps=1000,
            beta_schedule="linear",
            linear_start=1e-4,
            linear_end=2e-2,
            cosine_s=8e-3,

            loss_type="l2",
            log_every_t = 100,

            theta_dims = 9
    ):
        super().__init__()

        self.model = instantiate_from_config(diff_model_config)

        if loss_type not in ["l1", "l2", "huber"]:
            raise ValueError("__init__() got unknown loss type")
        self.loss_type = loss_type

        self.theta_dims = theta_dims

        # 每隔多少个t保持一次中间结果
        self.log_every_t = log_every_t

        # 预计算常量
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)       
       
    # 预计算alpha beta等常量和确定lvlb_weights
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # 前向过程
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        
        # 后向过程
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    # 正向过程, 计算p(xt|x0)
    # x_t = \sqrt{\hat{\beta_t}}*x_0 + \sqrt{(1 - \hat{\beta_t})} * z_t
    def q_sample(self, x, t, noise=None):
        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    # 基于L1，L2和smooth_L1度量模型预测结果和真实添加噪声之间的距离
    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(pred, target)
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # 计算batch样本loss: 采样噪声，得到x_t 预测噪声，度量距离
    def p_losses(self, x_start, t, cond=None, noise=None):
        '''
        :param x_start: x0, [batch_size, input_dim]
        :param t: 时间步t
        :param c: 生成条件
        :return: loss
        '''
        # 从高斯分布采样噪声\epsilon
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 得到x_t
        perturbed_x = self.q_sample(x_start, t, noise)
        # 网络预测\epsilon_\theta
        estimated_noise = self.model(perturbed_x, t, cond)
        # 度量距离
        loss = self.get_loss(estimated_noise, noise)
        return loss

    # 类入口，生成该batch对应的随机时间步t
    def forward(self, x, cond=None):
        b = x.shape[0]
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.p_losses(x, t, cond)

    # 逆向过程，基于模型预测得到的噪声 得到 x_{t-1}  p(x_{t-1}|x_t)
    @torch.no_grad()
    def p_sample(self, x, t, cond=None, biomarks=None):
        return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, cond, biomarks)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
        )

    # 逆向过程，整个反向去噪过程, 根据设置返回最终生成的x_0，或者保持中间量
    @torch.no_grad()
    def p_sample_loop(self, shape, device, cond=None, biomarks=None, return_intermediates=False):
        x = torch.randn(shape, device=device)
        batch_size = shape[0]
        diffusion_sequence = [x]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            # 对cond进行广播，匹配batch_size
            if cond is not None:
                num = batch_size // cond.shape[0]
                # print(num)
                cond = cond.repeat(int(num), 1, 1)
            # 对biomarks进行广播，匹配batch_size
            if biomarks is not None:
                num = batch_size // biomarks.shape[0]
                # print(num)
                biomarks = biomarks.repeat(int(num), 1)
            x = self.p_sample(x, t_batch, cond, biomarks)
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            if return_intermediates and (t % self.log_every_t == 0 or t == self.num_timesteps - 1):
                diffusion_sequence.append(x)

        if return_intermediates:
            return diffusion_sequence
        else:
            return x
        
    # 整体反向去噪过程入口，确定初始噪声shape
    @torch.no_grad()
    def sample(self, device="cuda", batch_size=16, cond=None,  return_intermediates=False):
        channels = self.theta_dims
        return self.p_sample_loop((batch_size, channels), device, cond, return_intermediates)




class ConditionalDiffusion(GaussianDiffusion):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 conditioning_key=None,
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 cosine_loss_weight=0,
                 *args, **kwargs):
        '''
        :param first_stage_config: 第一阶段投影编码器网络结构配置文件
        :param cond_stage_config: 第二阶段条件编码器网络结构配置文件
        :param cond_stage_key: 条件编码器对应的输入类型，对应如何处理输入
        :param concat_mode: 是否以concat作为条件模式
        :param conditioning_key: 如何使用条件模式
        :param cond_stage_trainable: 条件编码器是否可训练
        :param cond_stage_forward: 调用cond_stage哪个方法名作为入口
        :return: None
        '''
        super().__init__(*args, **kwargs)
        if conditioning_key is None:
            conditioning_key = 'crossattn'
        self.conditioning_key = conditioning_key
        self.cond_stage_trainable = cond_stage_trainable
        # self.cond_stage_key = cond_stage_key
        self.cond_stage_forward = cond_stage_forward
        # self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

        self.cosine_loss_weight = cosine_loss_weight

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        cond_model = model.cost if hasattr(model, "cost") else model
        self.cond_stage_model = cond_model

        if not self.cond_stage_trainable:
            self.cond_stage_model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            self.cond_stage_model.train()
            for param in self.cond_stage_model.parameters():
                param.requires_grad = True

    
    def get_first_stage_encoding(self, encoder_posterior):
        pass
    
    def get_learned_conditioning(self, cond):
        # 仅对 STMemCondEncoder 做输入维度修正: [B, T, C] -> [B, C, T]
        if cond is not None and hasattr(self, "cond_stage_model"):
            if self.cond_stage_model.__class__.__name__ == "STMemCondEncoder":
                if cond.ndim == 3 and cond.shape[1] == 2000 and cond.shape[2] == 1:
                    cond = cond.transpose(1, 2).contiguous()   # [B, 2000, 1] -> [B, 1, 2000]
        # print(f"the shape of cond = {cond.shape} in get_learned_conditioning")
        if self.cond_stage_forward is None:
            if self.cond_stage_trainable:
                if hasattr(self.cond_stage_model, 'cond_encode_grad') and callable(self.cond_stage_model.cond_encode_grad):
                    cond = self.cond_stage_model.cond_encode_grad(cond)
                else:
                    cond = self.cond_stage_model(cond)
            else:
                if hasattr(self.cond_stage_model, 'cond_encode') and callable(self.cond_stage_model.cond_encode):
                    cond = self.cond_stage_model.cond_encode(cond)
                else:
                    cond = self.cond_stage_model(cond)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            cond = getattr(self.cond_stage_model, self.cond_stage_forward)(cond)
        # print(f"the shape of cond = {cond.shape} in get_learned_conditioning")
        return cond


    # forword方法
    def forward(self, x, cond, biomarks, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        assert cond is not None
        cond = self.get_learned_conditioning(cond)
        # print(f"the shape of cond = {cond.shape}")
        return self.p_losses(x,  t, cond, biomarks,*args, **kwargs)
    
    # 得到扩散模型预测结果，根据self.conditioning_key处理输入x和条件
    def apply_model(self, x_noisy, t, cond, biomarks):
        assert self.conditioning_key in ['crossattn', 'hybrid']
        if self.conditioning_key == 'crossattn':
            out = self.model(input=x_noisy, t=t, cond=cond, biomarks=biomarks)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x_noisy] + [cond], dim=1)
            out = self.model(input=xc, t=t, cond=cond, biomarks=biomarks)
        else:
            raise NotImplementedError()
        return out
    
    # 计算batch样本loss: 采样噪声，基于self.apply_model()得到x_t 预测噪声，度量距离
    def p_losses(self, x_start, t, cond, biomarks, noise=None,*args, **kwargs):
        '''
        :param x_start: x0, [batch_size, input_dim]
        :param t: 时间步t
        :param cond: 生成条件
        :param biomarks: 手工特征条件
        :return: loss
        '''
        # 从高斯分布采样噪声\epsilon
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 得到x_t
        perturbed_x = self.q_sample(x_start, t, noise)
        # 网络预测\epsilon_\theta
        estimated_noise = self.apply_model(perturbed_x, t, cond, biomarks)

        # 度量距离
        L2_loss = self.get_loss(estimated_noise, noise)
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(estimated_noise, noise, dim=1)
        # 计算余弦损失
        cos_loss = 1 - cos_sim.mean()
        loss = L2_loss + self.cosine_loss_weight * cos_loss

        out = {}
        out['loss'] = loss
        return out

    @torch.no_grad()
    def p_sample(self, x, t, cond, biomarks):
        model_out = self.apply_model(x, t, cond, biomarks)
        return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * model_out) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
        )


    @torch.no_grad()
    def sample(self, device="cuda", batch_size=16, cond=None, biomarks = None, return_intermediates=False):
        assert exists(cond)
        cond = self.get_learned_conditioning(cond)
        channels = self.theta_dims
        return self.p_sample_loop((batch_size, channels), device, cond, biomarks, return_intermediates)

if __name__ == '__main__':
    pass