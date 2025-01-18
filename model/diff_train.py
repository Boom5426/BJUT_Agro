import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat

import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import sys
import os
from torch.optim.lr_scheduler import StepLR

from .diff_scheduler import NoiseScheduler
from process.utils import mask_tensor_with_masks
import torch.nn.functional as F


# 结合了均方误差（MSE）和绝对误差（MAE）的优点。它在误差较小时使用二次函数（MSE），在误差较大时使用线性函数（MAE），从而对异常值具有鲁棒性。
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        abs_error = torch.abs(input - target)
        is_small_error = abs_error <= self.delta
        small_error_loss = 0.5 * abs_error ** 2
        large_error_loss = self.delta * abs_error - 0.5 * self.delta ** 2
        return torch.where(is_small_error, small_error_loss, large_error_loss).mean()

# class diffusion_loss(nn.Module):
#     def __init__(self, penalty_factor=1.0, delta=1.0):
#         super(diffusion_loss, self).__init__()
#         self.mse = nn.MSELoss()
#         self.huber_loss = HuberLoss(delta=delta)
#         self.penalty_factor = penalty_factor

#     def forward(self, y_pred_0, y_true_0, y_pred_1, y_true_1):
#         loss_mse = self.mse(y_pred_0, y_true_0)
#         loss_huber = self.huber_loss(y_pred_1, y_true_1) * self.penalty_factor  # 计算 Huber 损失
#         return loss_mse + loss_huber


class diffusion_loss(nn.Module):
    def __init__(self, penalty_factor=1.0, delta=1.0, mse_weight=0.5, huber_weight=0.3, kl_weight=0.2):
        super(diffusion_loss, self).__init__()
        self.mse = nn.MSELoss()
        self.huber_loss = HuberLoss(delta=delta)
        self.penalty_factor = penalty_factor
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.kl_weight = kl_weight

    def forward(self, y_pred_0, y_true_0):
        # 计算各损失
        # loss_mse = self.mse(y_pred_0, y_true_0)
        # loss_huber = self.huber_loss(y_pred_0, y_true_0) * self.penalty_factor
        KL_loss = compute_kl_divergence(y_pred_0, y_true_0)

        # 加权求和
        # total_loss = (
        #     self.mse_weight * loss_mse +
        #     self.huber_weight * loss_huber +
        #     self.kl_weight * KL_loss
        # )
        return KL_loss


from torch.autograd import Variable
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

def compute_kl_divergence(tensor1, tensor2, reduction='batchmean'):
    """
    计算两个张量之间的 KL 散度。

    参数:
        tensor1 (torch.Tensor): 第一个张量，形状为 (batch_size, num_classes)。
        tensor2 (torch.Tensor): 第二个张量，形状为 (batch_size, num_classes)。
        reduction (str): 指定如何聚合 KL 散度。可选值为 'batchmean', 'mean', 'sum', 'none'。
            - 'batchmean': 对每个样本的 KL 散度取平均。
            - 'mean': 对所有样本的 KL 散度取平均。
            - 'sum': 对所有样本的 KL 散度求和。
            - 'none': 返回每个样本的 KL 散度。

    返回:
        torch.Tensor: 根据 reduction 参数返回的 KL 散度值。
    """
    # 确保输入是张量
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        raise ValueError("输入必须是 torch.Tensor 类型")

    # 对输入张量进行 softmax 归一化: 逐行转换
    prob1 = F.softmax(tensor1, dim=-1)
    prob2 = F.softmax(tensor2, dim=-1)

    # 计算 KL 散度
    kl_div = F.kl_div(prob1.log(), prob2, reduction='none').sum(dim=-1)

    # 根据 reduction 参数聚合结果
    if reduction == 'batchmean':
        return kl_div.mean()
    elif reduction == 'mean':
        return kl_div.mean()
    elif reduction == 'sum':
        return kl_div.sum()
    elif reduction == 'none':
        return kl_div
    else:
        raise ValueError("reduction 参数必须是 'batchmean', 'mean', 'sum' 或 'none'")


# def normal_train_diff(model,
#                  dataloader,
#                  lr: float = 1e-4,
#                  num_epoch: int = 1400,
#                  pred_type: str = 'noise',
#                  diffusion_step: int = 1000,
#                  device=torch.device('cuda:0'),
#                  is_tqdm: bool = True,
#                  is_tune: bool = False,
#                  mask_nonzero_ratio= None,
#                  mask_zero_ratio = None):
#     """通用训练函数

#     Args:
#         lr (float):
#         momentum (float): 动量
#         max_iteration (int, optional): 训练的 iteration. Defaults to 30000.
#         pred_type (str, optional): 预测的类型噪声或者 x_0. Defaults to 'noise'.
#         batch_size (int, optional):  Defaults to 1024.
#         diffusion_step (int, optional): 扩散步数. Defaults to 1000.
#         device (_type_, optional): Defaults to torch.device('cuda:0').
#         is_class_condi (bool, optional): 是否采用condition. Defaults to False.
#         is_tqdm (bool, optional): 开启进度条. Defaults to True.
#         is_tune (bool, optional): 是否用 ray tune. Defaults to False.
#         condi_drop_rate (float, optional): 是否采用 classifier free guidance 设置 drop rate. Defaults to 0..

#     Raises:
#         NotImplementedError: _description_
#     """

#     noise_scheduler = NoiseScheduler(
#         num_timesteps=diffusion_step,
#         beta_schedule='cosine'
#     )

#     criterion = diffusion_loss()  # loss_mse + loss_huber
#     # criterion = compute_kl_divergence()
#     model.to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
#     scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

#     if is_tqdm:
#         t_epoch = tqdm(range(num_epoch), ncols=100)
#     else:
#         t_epoch = range(num_epoch)

#     model.train()

#     for epoch in t_epoch:
#         epoch_loss = 0.
#         for i, (x, GE_CP_cond) in enumerate(dataloader): # 去掉了, celltype
#             x, GE_CP_cond = x.float().to(device), GE_CP_cond.float().to(device)
#             # print(x.shape, GE_cond.shape, CP_cond.shape) # torch.Size([159, 159]) torch.Size([159, 977]) torch.Size([159, 436])
#             # celltype = celltype.to(device)
#             # x, x_nonzero_mask, x_zero_mask = mask_tensor_with_masks(x, mask_zero_ratio, mask_nonzero_ratio)
#             # GE_cond, GE_cond_nonzero_mask, GE_cond_zero_mask = mask_tensor_with_masks(GE_cond, mask_zero_ratio, mask_nonzero_ratio)

#             x_noise = torch.randn(x.shape).to(device)
#             # GE_cond_noise = torch.randn(GE_cond.shape).to(device)

#             timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()
#             timesteps = timesteps.to(device)
#             x_t = noise_scheduler.add_noise(x,
#                                             x_noise,
#                                             timesteps=timesteps)

#             # GE_cond_t = noise_scheduler.add_noise(GE_cond,
#             #                                 GE_cond_noise,
#             #                                 timesteps=timesteps)

#             # mask = torch.tensor(mask).to(device)
#             # mask = (1-((torch.rand(x.shape[1]) < mask_ratio).int())).to(device)

#             # x_noisy = x_t + x
#             # x_noisy = x_t * x_nonzero_mask + x * (1 - x_nonzero_mask)
#             # GE_cond_noisy = GE_cond_t * GE_cond_nonzero_mask + GE_cond * (1 - GE_cond_nonzero_mask)

#             noise_pred = model(x_t, t=timesteps.to(device), GE_CP=GE_CP_cond) # 去掉了, z=celltype
#             # loss = criterion(noise_pred, noise)

#             loss = criterion(x_noise, noise_pred)
#             # print("loss:", x_noise.shape, noise_pred)  #  loss: torch.Size([159, 159]) tensor([[nan, nan, nan
#             # loss = criterion(x_noise * x_nonzero_mask, noise_pred * x_nonzero_mask, x_noise * x_zero_mask,
#             #                  noise_pred *  x_zero_mask)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
#             optimizer.step()
#             optimizer.zero_grad()
#             epoch_loss += loss.item()

#         scheduler.step()
#         epoch_loss = epoch_loss / (i + 1)  # type: ignore

#         current_lr = optimizer.param_groups[0]['lr']

#         # 更新tqdm的描述信息
#         if is_tqdm:
#             t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}, lr:{current_lr:.2e}')  # type: ignore

#         if is_tune:
#             session.report({'loss': epoch_loss})


def normal_train_diff(model,
                 dataloader,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 pred_type: str = 'noise',
                 diffusion_step: int = 1000,
                 device=torch.device('cuda:0'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 mask_nonzero_ratio= None,
                 mask_zero_ratio = None):
    """通用训练函数

    Args:
        lr (float):
        momentum (float): 动量
        max_iteration (int, optional): 训练的 iteration. Defaults to 30000.
        pred_type (str, optional): 预测的类型噪声或者 x_0. Defaults to 'noise'.
        batch_size (int, optional):  Defaults to 1024.
        diffusion_step (int, optional): 扩散步数. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:0').
        is_class_condi (bool, optional): 是否采用condition. Defaults to False.
        is_tqdm (bool, optional): 开启进度条. Defaults to True.
        is_tune (bool, optional): 是否用 ray tune. Defaults to False.
        condi_drop_rate (float, optional): 是否采用 classifier free guidance 设置 drop rate. Defaults to 0..

    Raises:
        NotImplementedError: _description_
    """

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    criterion = diffusion_loss()  # loss_mse + loss_huber
    # criterion = compute_kl_divergence()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.2)

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, GE_CP_cond) in enumerate(dataloader): # 去掉了, celltype
            x, GE_CP_cond = x.float().to(device), GE_CP_cond.float().to(device)
            # print(x.shape, GE_cond.shape, CP_cond.shape) # torch.Size([159, 159]) torch.Size([159, 977]) torch.Size([159, 436])

            x_noise = torch.randn(x.shape).to(device)
            # x_noise = torch.abs(torch.randn(x.shape).to(device))  # 使用绝对值确保非负的噪声
            # x_noise = torch.distributions.Exponential(rate=1.0).sample(x.shape).to(device)  # 使用指数分布的噪声

            timesteps = torch.randint(0, diffusion_step, (x.shape[0],)).long()
            timesteps = timesteps.to(device)
            x_t = noise_scheduler.add_noise(x,
                                            x_noise,
                                            timesteps=timesteps)

            noise_pred = model(x_t, t=timesteps.to(device), GE_CP=GE_CP_cond) # 去掉了, z=celltype
            # loss = criterion(noise_pred, noise)

            loss = criterion(x_noise, noise_pred)

            # x, x_cond = x.float().to(device), GE_CP_cond.float().to(device)
            # # celltype = celltype.to(device)
            # x, x_nonzero_mask, x_zero_mask = mask_tensor_with_masks(x, mask_zero_ratio, mask_nonzero_ratio)

            # x_noise = torch.randn(x.shape).to(device)

            # timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()
            # timesteps = timesteps.to(device)
            # x_t = noise_scheduler.add_noise(x,
            #                                 x_noise,
            #                                 timesteps=timesteps)

            # x_noisy = x_t * x_nonzero_mask + x * (1 - x_nonzero_mask)

            # noise_pred = model(x_noisy, t=timesteps.to(device), GE_CP=x_cond) # 去掉了, z=celltype
            # loss = criterion(x_noise * x_nonzero_mask, noise_pred * x_nonzero_mask, x_noise * x_zero_mask,
            #                  noise_pred *  x_zero_mask)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss = epoch_loss / (i + 1)  # type: ignore

        current_lr = optimizer.param_groups[0]['lr']

        # 更新tqdm的描述信息
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}, lr:{current_lr:.2e}')  # type: ignore

        if is_tune:
            session.report({'loss': epoch_loss})
