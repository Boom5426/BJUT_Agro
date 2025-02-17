import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import joblib
import yaml
import os
import pandas as pd
import warnings
import torch
import json
import heapq
import yaml
import joblib
import argparse
from model.diff_model import DiT_diff
from model.DiT import DiT
from model.diff_scheduler import NoiseScheduler
from model.diff_train import normal_train_diff
from model.sample import sample_diff
from process.utils import *
from process.data import *
from process.evaluation import *
import warnings
warnings.filterwarnings("ignore")
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--data_path", type=str, default='datasets/BJUT_all.npy')
# parser.add_argument("--data_path", type=str, default='datasets/subLINCS_train_test.npy') #　subLINCS_train.csv

parser.add_argument("--document", type=str, default='1_10')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=1024)  # 2048
parser.add_argument("--hidden_size", type=int, default=512)  # 512
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--diffusion_step", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-4)  # 太高了容易梯度爆炸
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--noise_std", type=float, default=10)
parser.add_argument("--pca_dim", type=int, default=100)
parser.add_argument("--head", type=int, default=16)
parser.add_argument("--mask_nonzero_ratio", type=float, default=0.3)
parser.add_argument("--mask_zero_ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=3407)
args = parser.parse_args()

print(os.getcwd())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
# 读取数据函数
def read_data(file_path):
    """
    读取txt文件中的数据，并返回为NumPy数组
    """
    data = np.loadtxt(file_path)
    return data

# 计算MSE、MAPE、PCC、R2、MAE
def calculate_metrics(y_true, y_pred):
    """
    计算MSE、MAPE、PCC、R2、MAE指标
    """
    # MSE (Mean Squared Error)
    mse = mean_squared_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # PCC (Pearson Correlation Coefficient)
    pcc, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    
    # R2 (R-Squared)
    r2 = r2_score(y_true, y_pred)
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    
    return mse, mape, pcc, r2, mae

# 主程序
def train_valid_test():
    """
    训练、验证和测试过程，包括模型训练和生成预测
    """
    # 设定文件路径和配置
    seed_everything(args.seed)
    data_path = args.data_path
    directory = 'save/' + args.document + '_ckpt/'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, args.document + '.pt')

    # 数据加载和分割
    dataset = ConditionalDiffusionDataset(data_path)
    (train_dataset, train_smiles_names), (test_dataset, test_smiles_names) = split_dataset_with_smiles_names(dataset, train_ratio=0.9, random_state=42)

    print("Split train : test", len(train_smiles_names), len(test_smiles_names))  # 输出训练和测试集大小
    
    # 数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型定义
    model = DiT_diff(
        st_input_size=10, 
        condi_GE_CP_size=14927,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.head,
        classes=6,
        mlp_ratio=4.0,
        pca_dim=args.pca_dim,
        dit_type='dit'
    )
    model.to(args.device)
    diffusion_step = args.diffusion_step

    # 训练过程
    model.train()
    normal_train_diff(model,
                        dataloader=train_dataloader,
                        lr=args.learning_rate,
                        num_epoch=args.epoch,
                        diffusion_step=diffusion_step,
                        device=args.device,
                        pred_type='noise',
                        mask_nonzero_ratio=args.mask_nonzero_ratio,
                        mask_zero_ratio=args.mask_zero_ratio)
    torch.save(model.state_dict(), save_path)

    # 测试过程
    noise_scheduler = NoiseScheduler(num_timesteps=diffusion_step, beta_schedule='cosine')
    with torch.no_grad():
       test_gt = torch.stack([data for data, _ in test_dataset])
       test_GE_CP = torch.stack([GE_CP for _, GE_CP in test_dataset])

    print(test_gt.shape, test_GE_CP.shape)  # 输出测试集的形状
    
    prediction = sample_diff(model,
                            device=args.device,
                            dataloader=test_dataloader,
                            noise_scheduler=noise_scheduler,
                            mask_nonzero_ratio=0.3,
                            mask_zero_ratio=0,
                            gt=test_gt,
                            GE_CP=test_GE_CP,
                            num_step=diffusion_step,
                            sample_shape=(test_gt.shape[0], test_gt.shape[1]),
                            is_condi=True,
                            sample_intermediate=diffusion_step,
                            model_pred_type='x_start',
                            is_classifier_guidance=False,
                            omega=0.9
                            )

    return prediction, test_gt, train_smiles_names

# 设置文件路径和目录
Data = args.document
outdir = 'save/' + Data + '_ckpt/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

hyper_directory = outdir
hyper_file = Data + '_hyperameters.yaml'
hyper_full_path = os.path.join(hyper_directory, hyper_file)

if not os.path.exists(hyper_directory):
    os.makedirs(hyper_directory)

# 保存超参数设置
args_dict = vars(args)
with open(hyper_full_path, 'w') as yaml_file:
    yaml.dump(args_dict, yaml_file)

# 训练并得到预测结果和真实标签
prediction_result, ground_truth, train_smiles_names = train_valid_test()

# 反归一化操作
scaler_smiles = joblib.load('./datasets/scaler_smiles.pkl')
prediction_result_smiles = torch.tensor(scaler_smiles.inverse_transform(prediction_result))
ground_truth_smiles = torch.tensor(scaler_smiles.inverse_transform(ground_truth.numpy()))

# 保存预测结果和真实标签到文件
np.savetxt(outdir + '/Drug_prediction2.txt', prediction_result_smiles, delimiter=' ')
np.savetxt(outdir + '/Drug_GT2.txt', ground_truth_smiles, delimiter=' ')

# 计算评价指标
print("\n########### eavling #############\n")

# 读取生成的文件
true_file_path = outdir + "Drug_GT2.txt"  # 真实值文件路径
pred_file_path = outdir + "Drug_prediction2.txt"  # 预测值文件路径

y_true = read_data(true_file_path)
y_pred = read_data(pred_file_path)

# 检查数据形状是否一致
if y_true.shape != y_pred.shape:
    raise ValueError("真实值和预测值的形状不一致！")

# 计算各项评价指标
mse, mape, pcc, r2, mae = calculate_metrics(y_true, y_pred)

# 输出结果
print(f"MSE: {mse:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"PCC: {pcc:.4f}")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")

