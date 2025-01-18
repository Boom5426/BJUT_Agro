import pandas as pd
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
parser.add_argument("--batch_size", type=int, default=1000)  # 2048
parser.add_argument("--hidden_size", type=int, default=512)  # 512
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--diffusion_step", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4)  # 太高了容易梯度爆炸
parser.add_argument("--depth", type=int, default=2)
parser.add_argument("--noise_std", type=float, default=10)
parser.add_argument("--pca_dim", type=int, default=100)
parser.add_argument("--head", type=int, default=16)
parser.add_argument("--mask_nonzero_ratio", type=float, default=0.3)
parser.add_argument("--mask_zero_ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=3407)
args = parser.parse_args()

print(os.getcwd())
print(torch.cuda.get_device_name(torch.cuda.current_device()))


def train_valid_test():
    seed_everything(args.seed)

    data_path = args.data_path

    directory = 'save/' + args.document + '_ckpt/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, args.document + '.pt')

    dataset = ConditionalDiffusionDataset(data_path)
    # dataset = ConditionalDiffusion_LInCS(data_path)  #　ConditionalDiffusion_LInCS
    (train_dataset, train_smiles_names), (test_dataset, test_smiles_names) = split_dataset_with_smiles_names(dataset, train_ratio=0.9, random_state=42)

    # # 指定训练和测试数据集    
    # train_dataset = ConditionalDiffusion_LInCS(data_path)  #　ConditionalDiffusion_LInCS
    # test_dataset = ConditionalDiffusion_LInCS(data_path)  #　ConditionalDiffusion_LInCS

    print("Split train : test",len(train_smiles_names), len(test_smiles_names)) # 16276 4070  smiles names = smiles?
    # all_data_matrix = torch.stack([data for data, _ in valid_dataset])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # cell_num = dataset.smiles_names.shape[0]
    # spot_num = dataset.smiles_names.shape[1]
    # print("cell_num", cell_num, spot_num) # 26252 41674
    # mask_1 = (1 - ((torch.rand(st_smiles_num) < args.mask_ratio).int())).to(args.device)
    # mask_0 = (1 - ((torch.rand(st_smiles_num) < args.mask_ratio).int())).to(args.device)

    model = DiT_diff( # DiT
    # model = DiT( # DiT
        st_input_size=10, # 分子维度： 159 for cpg0003, 279 for LINCS
        condi_GE_CP_size=14927,  # gene + CP 的维度: 977 for cpg0003, 978 for LINCS, 1413 for all
        # condi_CP_size=436,  # gene 的维度
        hidden_size=args.hidden_size,  # 256
        depth=args.depth,
        num_heads=args.head,
        classes=6,
        mlp_ratio=4.0,
        pca_dim=args.pca_dim,
        dit_type='dit'
    )

    model.to(args.device)
    diffusion_step = args.diffusion_step

    model.train()

    # model.load_state_dict(torch.load(save_path))
    
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

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )
    with torch.no_grad():
       test_gt = torch.stack([data for data, _ in test_dataset])
       test_GE_CP = torch.stack([GE_CP for _, GE_CP in test_dataset])
    #    test_CP = torch.stack([CP for _, _, CP in test_dataset])
       print(test_gt.shape, test_GE_CP.shape) # torch.Size([4070, 159]) torch.Size([4070, 977]) torch.Size([4070, 436])
       # test_gt = torch.randn(len(test_dataset), 249)
       prediction = sample_diff(model,
                                device=args.device,
                                dataloader=test_dataloader,
                                noise_scheduler=noise_scheduler,
                                mask_nonzero_ratio=0.3,
                                mask_zero_ratio = 0,
                                gt=test_gt,
                                GE_CP=test_GE_CP,
                                # CP=test_CP,
                                num_step=diffusion_step,
                                sample_shape=(test_gt.shape[0], test_gt.shape[1]),
                                is_condi=True,
                                sample_intermediate=diffusion_step,
                                model_pred_type='x_start',
                                is_classifier_guidance=False,
                                omega=0.9
                                )

    return prediction, test_gt, train_smiles_names


Data =  args.document
outdir = 'save/'+Data+'_ckpt/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

hyper_directory = outdir
hyper_file = Data + '_hyperameters.yaml'
hyper_full_path = os.path.join(hyper_directory, hyper_file)
if not os.path.exists(hyper_directory):
    os.makedirs(hyper_directory)
args_dict = vars(args)
with open(hyper_full_path, 'w') as yaml_file:
    yaml.dump(args_dict, yaml_file)

prediction_result, ground_truth, train_smiles_names = train_valid_test()

# 反归一化
# 加载归一化器
scaler_smiles = joblib.load('./datasets/scaler_smiles.pkl')

prediction_result_smiles = torch.tensor(scaler_smiles.inverse_transform(prediction_result))
ground_truth_smiles = torch.tensor(scaler_smiles.inverse_transform(ground_truth.numpy()))

np.savetxt(outdir + '/Drug_prediction.txt', prediction_result_smiles, delimiter=' ')
np.savetxt(outdir + '/Drug_GT.txt', ground_truth_smiles, delimiter=' ')


# np.savetxt(outdir + '/Drug_prediction.txt', prediction_result, delimiter=' ')
# np.savetxt(outdir + '/Drug_GT.txt', ground_truth, delimiter=' ')



print("\n########### Testing Performance #############\n")
# 根据生成的 txt 文件来测试性能
# 读取txt中的每一行，如　“1 5 0 0 9 5 9 5 46”，然后拆分，根据字典“./dataset/class_to_number.json”映射到原始的键值，输出selfies，转为smiles。 

#　转为 smiles
with open("./datasets/class_to_number_LINCS.json", "r") as file:
    class_to_number = json.load(file)
number_to_class = {v: k for k, v in class_to_number.items()}

def id2seq(ID, ID_Dict):
    seq = ''
    for i in ID:
        if i in [0, 1, 2]:  # 跳过 0, 1, 2
            continue
        seq += ID_Dict.get(i, '')
    return seq

# 读取txt文件并处理每一行
with open(outdir + "Drug_prediction.txt", "r") as file:
    lines = file.readlines()

# print(class_to_number)  #　{'[#Branch1]': 3, '[Branch2]': 4, '[C]': 5, 

# 处理每一行
Result_all = []
smiles_list = []
for line in lines:
    ids = list(map(int, line.strip().split()))
    selfies_seq = id2seq(ids, number_to_class)
    smiles = sf.decoder(selfies_seq)
    smiles_list.append(smiles)
    # print(smiles)
    # break

out_valid, out_unval_index = valid(smiles_list) # 计算方式有问题吧？
# print(out_valid)
out_valid_list = [smiles_list[j] for j in range(len(smiles_list)) if j not in out_unval_index]
out_set, out_unique = unique(out_valid_list)  # 唯一性
# print("222")
set_gen = set(out_valid_list) #　独特性是针对训练集中的分子来说的
ref_smiles = train_smiles_names
ref_smiles_set = set(ref_smiles)
novelty = 1-(len(set_gen.intersection(ref_smiles_set)) / len(out_valid_list))

out_div = IntDivp(out_set) # 计算生成分子集合的内部多样性（Internal Diversity, IntDiv
IntDiv = np.mean(out_div)

mean_qed, top10_qed, QED_scores = Qed(out_set)
with open(outdir + 'QED_scores.txt', 'w') as f:
    for score in QED_scores:
        f.write(f"{score}\n")

mean_SA, top_n_SA, SA_scores = SA(out_set)
# 将 SA_scores 保存为 TXT 文件
with open(outdir + 'SA_scores.txt', 'w') as f:
    for score in SA_scores:
        f.write(f"{score}\n")

result_lines = [
    f"Valid: {out_valid:.3f};",
    f"Novelty: {novelty:.3f};",
    f"Unique: {out_unique:.3f};",
    f"IntDiv: {IntDiv:.3f};",
    f"Mean_QED: {mean_qed:.3f};",
    f"Top10_QED: {top10_qed:.3f};",
    f"Mean_SA: {mean_SA:.3f};",
    f"Top10_SA: {top_n_SA:.3f};",
]

# 将结果写入文本文件
with open(outdir + "test_performance.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(result_lines))

# # 将结果添加到 Result_all 列表中
# Result_all.append(Result_line)

# # 保存测试结果
# df = pd.DataFrame(Result_all, columns=["Valid", "Novelty", "Unique", "IntDiv", "Mean_QED", "Top10_QED", "Mean_SA", "Top10_SA"])

# # 将所有列的值格式化为 "指标名: 数值"
# for col in df.columns:
#     df[col] = df[col].apply(lambda x: f"{col}: {x.split(': ')[1]}")

# # 保存为 CSV 文件
# df.to_csv(outdir + "test_performance.csv", index=False)

print("Finish!", "results in", outdir+"test_performance.txt")

'''
QED（Quantitative Estimate of Drug-likeness）: QED 是一个定量评估分子药物相似性的指标，范围在0到1之间。QED 值越高，表示分子越有可能具有药物类似的特性。
SA（Synthetic Accessibility score）: SA 是一个评估分子合成难易程度的指标，通常范围在1到10之间。SA 值越低，表示分子越容易合成
'''