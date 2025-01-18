import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from process.utils import calculate_rmse_per_gene, calculate_pcc_per_gene,calculate_pcc_with_mask,calculate_rmse_with_mask
from process.utils import mask_tensor_with_masks

def model_sample_diff(model, device, dataloader, x, time, is_condi, condi_flag):
    noise = []
    i = 0
    for _, GE_CP in dataloader: # 计算整个shape得噪声 一次循环算batch大小
        # print(i, GE_CP.shape) #　0 torch.Size([1000, 977]); 1000 torch.Size([1000, 977])
        GE_CP = GE_CP.float().to(device) # x.float().to(device)
        t = torch.full((GE_CP.shape[0],), time, dtype=torch.long, device=device)
        # t = torch.from_numpy(np.repeat(time, GE_CP.shape[0])).long().to(device)
        # celltype = celltype.to(device)
        if not is_condi:
            n = model(x, t, None)
        else: # 进来
            n = model(x[i:i+len(GE_CP)], t, GE_CP, condi_flag=condi_flag)
        noise.append(n)
        i = i+len(GE_CP)
    noise = torch.cat(noise, dim=0)
    return noise

def sample_diff(model,
                dataloader,
                noise_scheduler,
                mask_nonzero_ratio = None,
                mask_zero_ratio = None,
                gt = None,
                GE_CP = None,
                # CP = None,
                device=torch.device('cuda:0'),
                num_step=1000,
                sample_shape=(4700, 159),
                is_condi=False,
                sample_intermediate=200,
                model_pred_type: str = 'noise',
                is_classifier_guidance=False,
                omega=0.1,
                is_tqdm = True):
    model.eval()
    gt = torch.tensor(gt).to(device)
    GE_CP = torch.tensor(GE_CP).to(device)
    # CP = torch.tensor(CP).to(device)

    # 指定噪声,和前面训练保持一致
    x_t = torch.randn(sample_shape[0], sample_shape[1]).to(device)
    # x_t = torch.abs(torch.randn(sample_shape[0], sample_shape[1]).to(device))  # 使用绝对值确保非负的噪声
    # x_t = torch.distributions.Exponential(rate=1.0).sample(sample_shape).to(device)  # 使用指数分布的噪声
    # x_noise = torch.distributions.Exponential(rate=1.0).sample(x.shape).to(device)  # 使用指数分布的噪声
    
    timesteps = list(range(num_step))[::-1]  # 倒序
    # gt_mask, mask_nonzero, mask_zero = mask_tensor_with_masks(gt, mask_zero_ratio, mask_nonzero_ratio)
    # mask = torch.tensor(mask_nonzero).to(device)
    # mask = None
    # x_t =  x_t * (1 - mask) + gt * mask
    # x_t = x_t  + gt * mask
    x_t = x_t
    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]

    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            # 输出噪声
            model_output = model_sample_diff(model,
                                        device=device,
                                        dataloader=dataloader,
                                        x=x_t,  # x_t
                                        time=time,  # t
                                        is_condi=is_condi,
                                        condi_flag=True)
            if is_classifier_guidance:
                model_output_uncondi = model_sample_diff(model,
                                                    device=device,
                                                    dataloader=dataloader,
                                                    x=x_t,
                                                    time=time,
                                                    is_condi=is_condi,
                                                    condi_flag=False)
                model_output = (1 + omega) * model_output - omega * model_output_uncondi

        # 计算x_{t-1}
        x_t, _ = noise_scheduler.step(model_output,  # 一般是噪声
                                     torch.from_numpy(np.array(time)).long().to(device),
                                      x_t, model_pred_type=model_pred_type)

        # if mask is not None:
        #     x_t = x_t *  mask + (1 - mask) * gt # 其实是用 GT 过滤了一遍，这个公平吗？

        if time == 0 and model_pred_type == 'x_start':
            # 如果直接预测 x_0 的话，最后一步直接输出
            sample = model_output


    recon_x = x_t.detach().cpu().numpy()
    
    return recon_x

