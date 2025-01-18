import scipy
import scanpy as sc
import numpy as np
import torch
import pandas as pd
import joblib
from torch.utils.data import TensorDataset, DataLoader, Dataset
from anndata import AnnData
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
CHUNK_SIZE = 20000

# 处理数据：去除 '//' 分隔符并转换为浮点数
def process_column(data):
    # 将每个元素按 '//' 分割，并转换为浮点数
    processed_data = [float(value) for value in data.split('//') if value]
    return processed_data
class ConditionalDiffusion_LInCS(Dataset):
    def __init__(self, data_path, normalize=True, scaler_type='standard'):
        self.all_data = np.load(data_path, allow_pickle=True)
        
        # 提取 smiles 数据并转换为张量
        self.smiles = torch.tensor([seq for seq in self.all_data[:, 1]], dtype=torch.float32)  # (20346, 159)

        self.column_7 = self.all_data[:, 6]

        processed_column_7 = [process_column(item) for item in self.column_7]
        # 将处理后的数据转换为 PyTorch 张量
        self.GE_data = torch.tensor(processed_column_7, dtype=torch.float32)

        # 归一化处理
        self.normalize = normalize
        if self.normalize:
            if scaler_type == 'minmax':
                self.scaler_smiles = MinMaxScaler()
                # self.scaler_GE = MinMaxScaler()
            elif scaler_type == 'standard':
                self.scaler_smiles = StandardScaler()
                # self.scaler_GE = StandardScaler()
            else:
                raise ValueError("scaler_type must be 'minmax' or 'standard'")
            
            # 对 smiles 和 GE_data 进行归一化
            self.smiles = torch.tensor(self.scaler_smiles.fit_transform(self.smiles), dtype=torch.float32)
            # self.GE_data = torch.tensor(self.scaler_GE.fit_transform(self.GE_data), dtype=torch.float32)
            # 保存归一化器到文件
            joblib.dump(self.scaler_smiles, './datasets/scaler_smiles.pkl')

        print(self.GE_data.shape, self.smiles.shape)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.smiles[idx], self.GE_data[idx]
    
    def get_smiles_names(self):
        return self.smiles



class ConditionalDiffusionDataset(Dataset):
    def __init__(self, data_path, normalize=True, scaler_type='minmax'):
        self.all_data = np.load(data_path, allow_pickle=True)

        # 提取数据并转换为 float32
        smiles_data = self.all_data[:, 18:28].astype(np.float32)
        # 转换为 PyTorch 张量
        self.smiles = torch.tensor(smiles_data, dtype=torch.float32)

        # self.smiles = torch.tensor(self.all_data[:, 18:28], dtype=torch.float32)
        # # 数据填充，nan to 0, 不然loss一直是nan
        # self.GE_data = torch.nan_to_num(self.GE_data, nan=0.0)

        GE_data = self.all_data[:, 45:].astype(np.float32)
        # 转换为 PyTorch 张量
        # self.smiles = torch.tensor(smiles_data, dtype=torch.float32)
        self.GE_data = torch.tensor(GE_data, dtype=torch.float32)  # 977 gene

        # # 归一化处理
        self.normalize = normalize
        if self.normalize:
            if scaler_type == 'minmax':
                self.scaler_smiles = MinMaxScaler()
                self.scaler_GE = MinMaxScaler()
            elif scaler_type == 'standard':
                self.scaler_smiles = StandardScaler()
                self.scaler_GE = StandardScaler()
            else:
                raise ValueError("scaler_type must be 'minmax' or 'standard'")
            
            # 对 smiles 和 GE_data 进行归一化
            self.smiles = torch.tensor(self.scaler_smiles.fit_transform(self.smiles), dtype=torch.float32)
            self.GE_data = torch.tensor(self.scaler_GE.fit_transform(self.GE_data), dtype=torch.float32)
            # 保存归一化器到文件
            joblib.dump(self.scaler_smiles, './datasets/scaler_smiles.pkl')

        print(self.GE_data.shape, self.smiles.shape)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.smiles[idx], self.GE_data[idx]
    
    def get_smiles_names(self):
        return self.smiles


def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list

    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing
    chunk_size
        chunk large data into small chunks

    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        for i in range(new_X.shape[0] // chunk_size + 1):
            new_X[i * chunk_size:(i + 1) * chunk_size, idx] = adata[i * chunk_size:(i + 1) * chunk_size, genes[idx]].X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={'var_names': genes})
    return adata


def plot_hvg_umap(hvg_adata, color=['celltype'], save_filename=None):
    sc.set_figure_params(dpi=80, figsize=(3, 3))  # type: ignore
    hvg_adata = hvg_adata.copy()
    if save_filename:
        sc.settings.figdir = save_filename
        save = '.pdf'
    else:
        save = None
    # ideal gas equation

    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata)
    sc.pp.neighbors(hvg_adata, n_pcs=30, n_neighbors=30)
    sc.tl.umap(hvg_adata, min_dist=0.1)
    sc.pl.umap(hvg_adata, color=color, legend_fontsize=10, ncols=2, show=None, save=save, wspace=1)
    return hvg_adata


def get_data_loader(data_ary: np.ndarray,
                    cell_type: np.ndarray,
                    batch_size: int = 512,
                    is_shuffle: bool = True,
                    ):
    data_tensor = torch.from_numpy(data_ary.astype(np.float32))
    cell_type_tensor = torch.from_numpy(cell_type.astype(np.float32))
    dataset = TensorDataset(data_tensor, cell_type_tensor)
    generator = torch.Generator(device='cuda')
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False,
        generator=generator)  # , generator=torch.Generator(device = 'cuda')


def scale(adata):
    scaler = MaxAbsScaler()
    # 对adata.X按行进行归一化
    normalized_data = scaler.fit_transform(adata.X.T).T

    # 更新归一化后的数据到adata.X
    adata.X = normalized_data
    return adata


def data_augment(adata: AnnData, fixed: bool, noise_std):
    # 定义增强参数，例如噪声的标准差
    noise_stddev = noise_std
    augmented_adata = adata.copy()
    gene_expression = adata.X

    if fixed:
        augmented_adata.X = augmented_adata.X + np.full(gene_expression.shape, noise_stddev)
    else:
        # 对每个基因的表达值引入随机噪声
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_stddev, gene_expression.shape))

    merge_adata = adata.concatenate(augmented_adata, join='outer')

    return merge_adata




