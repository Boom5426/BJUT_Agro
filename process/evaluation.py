import numpy as np
import pandas as pd
from tqdm import tqdm
import selfies as sf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle
from six import iteritems
import math
from collections import defaultdict
import os.path as op
from rdkit import rdBase, Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fraggle import FraggleSim
from rdkit.Chem.Scaffolds import MurckoScaffold
import seaborn as sns
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger

# 禁用 RDKit 日志
RDLogger.DisableLog('rdApp.*')

def id2seq(ID, ID_Dict):
    seq=''
    print(ID, ID_Dict)
    for i in ID[1:]:  # 1开头，2结尾，0填充
        if i == 2: break
        seq += ID_Dict[i]
    
    return sf.decoder(seq)  #　转了

def valid(l): 
    val = 0
    unval_index = []
    for i in range(len(l)):
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(l[i]), isomericSmiles=True)
            val += 1
        except:
            # print("not successfully processed smiles: ", l[i])
            unval_index.append(i)
            pass
    valid = val/len(l)
    return valid, unval_index

def unique(l):
    l_set = list(set(l)) # 使用 set(l) 去除列表中的重复分子，得到唯一的分子集合 l_set。
    uniq = len(l_set)/len(l) # 计算唯一性分数 uniq，即唯一分子数量与总分子数量的比值。
    return l_set, uniq

def IntDivp(out):
    all_div = []
    for i in range(len(out)-1):
        div = 0.0
        tot = 0
        for j in range(i+1, len(out)):
            ms = [Chem.MolFromSmiles(out[i]), Chem.MolFromSmiles(out[j])]
            mfp = [Chem.RDKFingerprint(x) for x in ms]
            div += 1 - DataStructs.FingerprintSimilarity(mfp[0], mfp[1], metric=eval(metic_list[0]))
            tot +=1
        div /= tot
        all_div.append(div)
    all_div = np.array(all_div)
    return all_div

# def Qed(out_set):
#     qed_list = []
#     qed_scores = []
#     for i in out_set:
#         mol = Chem.MolFromSmiles(i)
#         qed_score = QED.default(mol)
#         qed = QED.properties(mol)
#         qed_scores.append(qed_score)
#         qed_list.append(qed)
#     return qed_list, qed_scores

def Qed(out_set):
    qed_scores = []
    for i in out_set:
        mol = Chem.MolFromSmiles(i)
        if mol is not None:  # 确保分子有效
            qed_score = QED.default(mol)
            qed_scores.append(qed_score)
    
    # 计算 QED 分数的均值
    mean_qed = np.mean(qed_scores) if qed_scores else 0.0
    
    # 计算前 10 个 QED 分数的均值
    top10_qed = np.mean(sorted(qed_scores, reverse=True)[:10]) if qed_scores else 0.0
    
    return mean_qed, top10_qed, qed_scores

_fscores = None
def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    _fscores = pickle.load(gzip.open('%s.pkl.gz'%name))
    outDict = {}
    for i in _fscores:
        for j in range(1,len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol,ri=None):
  nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
  nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
  return nBridgehead,nSpiro

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
def calculateScore(m):
    if _fscores is None: readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,2)  #<- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():  # 使用 items() 代替 iteritems()
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf
    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m,includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads,nSpiro=numBridgeheadsAndSpiro(m,ri)
    nMacrocycles=0
    for x in ri.AtomRings():
        if len(x)>8: nMacrocycles+=1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters+1)
    spiroPenalty = math.log10(nSpiro+1)
    bridgePenalty = math.log10(nBridgeheads+1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0: macrocyclePenalty = math.log10(2)

    score2 = 0. -sizePenalty -stereoPenalty -spiroPenalty -bridgePenalty -macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.: sascore = 8. + math.log(sascore+1.-9.)
    if sascore > 10.: sascore = 10.0
    elif sascore < 1.: sascore = 1.0 

    return sascore
    

def processMols(mols):
  print('smiles\tName\tsa_score')
  for i,m in enumerate(mols):
    if m is None:
      continue
 
    s = calculateScore(m)

    smiles = Chem.MolToSmiles(m)
    print(smiles+"\t"+m.GetProp('_Name') + "\t%3f"%s)


if __name__=='__main__':
  import sys,time

  t1=time.time()
  readFragmentScores("fpscores")
  t2=time.time()

  suppl = Chem.SmilesMolSupplier(sys.argv[1])
  t3=time.time()
  processMols(suppl)
  t4=time.time()

  print('Reading took %.2f seconds. Calculating took %.2f seconds'%((t2-t1),(t4-t3)), file=sys.stderr)


def SA(out_set, top_n=10):
    """
    计算分子集合中每个分子的 SA 分数，并返回均值及前 N 个 SA 分数的均值。
    
    参数:
        out_set (list): 分子 SMILES 字符串列表。
        top_n (int): 计算前 N 个 SA 分数的均值，默认为 10。
    
    返回:
        mean_SA (float): 所有 SA 分数的均值。
        top_n_SA (float): 前 N 个 SA 分数的均值。
    """
    SA_scores = []
    for i in out_set:
        mol = Chem.MolFromSmiles(i)
        if mol is not None:  # 确保分子有效
            SA_score = calculateScore(mol)
            SA_scores.append(SA_score)
    
    # 计算均值
    mean_SA = np.mean(SA_scores) if SA_scores else 0.0
    
    # 计算前 N 个 SA 分数的均值
    if len(SA_scores) >= top_n:
        top_n_SA = np.mean(sorted(SA_scores)[:top_n])  # SA 分数越低越好
    else:
        top_n_SA = np.mean(SA_scores) if SA_scores else 0.0
    
    return mean_SA, top_n_SA, SA_scores

# def SA(out_set):
#     SA_scores = []
#     for i in out_set:
#         mol = Chem.MolFromSmiles(i)
#         SA_score = calculateScore(mol)
#         SA_scores.append(SA_score)
#     return SA_scores

def Fragment_similarity(list_p, list_t):
    print("输入顺序：predict_list, true_list")
    fraggle_similarity = []
    smi_frag = []
    for i in tqdm(range(len(list_t))):
        mol1 = Chem.MolFromSmiles(list_p[i])
        mol2 = Chem.MolFromSmiles(list_t[i])
        try:
            (smi, match) = FraggleSim.GetFraggleSimilarity(mol1,mol2)
            fraggle_similarity.append(smi)
            smi_frag.append(match)
        except:
            fraggle_similarity.append(0)
            smi_frag.append("None")
    return fraggle_similarity, smi_frag

def Fragment_similarity_(list_p, list_n):
    print("输入顺序：predict_list, true_list")
    fraggle_similarity = []
    sim_frag = []
    for i in tqdm(range(len(list_p))):
        a,b=[],[]
        mol1 = Chem.MolFromSmiles(list_p[i])
        for j in range(len(list_n)):
            mol2 = Chem.MolFromSmiles(list_n[j])
            try:
                (smi, match) = FraggleSim.GetFraggleSimilarity(mol1,mol2)
                a.append(smi)
                b.append(match)
            except:
                a.append(0)
                b.append("None")
        fraggle_similarity.append(a)
        sim_frag.append(b)
    return fraggle_similarity, sim_frag

def Scaffold_similarity(list_p, list_t):
    print("输入顺序：predict_list, true_list")
    scaff_sim = []
    for i in range(len(list_p)):
        mol1_scaff = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(list_p[i]))
        mol2_scaff = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(list_t[i]))
        mfp = [Chem.RDKFingerprint(x) for x in [mol1_scaff, mol2_scaff]]
        smi = DataStructs.FingerprintSimilarity(mfp[0], mfp[1], metric=eval(metic_list[2]))
        scaff_sim.append(smi)
    return scaff_sim

def Scaffold_similarity_(list_p, list_n):
    print("输入顺序：predict_list, nsclc_list")
    scaff_sim = []
    for i in tqdm(range(len(list_p))):
        a=[]
        mol1_scaff = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(list_p[i]))
        for j in range(len(list_n)):
            mol2_scaff = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(list_n[j]))                
            mfp = [Chem.RDKFingerprint(x) for x in [mol1_scaff, mol2_scaff]]
            smi = DataStructs.FingerprintSimilarity(mfp[0], mfp[1], metric=eval(metic_list[2]))
            a.append(smi)
        scaff_sim.append(a)
    return scaff_sim

def ECFP_Tanimoto_similarity_(list_p, list_n):
    print("输入顺序：predict_list, nsclc_list")
    out=[]
    for i in tqdm(range(len(list_p))):
        mol1 = Chem.MolFromSmiles(list_p[i])
        mol2 = Chem.MolFromSmiles(list_n[j])
        ECFPs = [Chem.AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in [mol1, mol2]]
        out.append(DataStructs.FingerprintSimilarity(ECFPs[0], ECFPs[1], metric=eval(metic_list[0])))
    return out

def ECFP_Tanimoto_similarity_(list_p, list_n):
    print("输入顺序：predict_list, nsclc_list")
    out=[]
    for i in tqdm(range(len(list_p))):
        a=[]
        mol1 = Chem.MolFromSmiles(list_p[i])
        for j in range(len(list_n)):
            mol2 = Chem.MolFromSmiles(list_n[j])
            ECFPs = [Chem.AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in [mol1, mol2]]
            a.append(DataStructs.FingerprintSimilarity(ECFPs[0], ECFPs[1], metric=eval(metic_list[0])))
        out.append(a)
    return out

def Fingerprint_similarity(list_p, list_t):
    print("输入顺序：predict_list, true_list")
#     RDKfp_sim,MACCS_sim,AP_sim,tts_sim,MGfp_sim,ECFP4_sim,FCFP4_sim = [],[],[],[],[],[],[]
    RDKfp_sim, MACCS_sim, MGfp_sim = [], [], []
    for i in range(len(list_p)):
        mol1 = Chem.MolFromSmiles(list_p[i])
        mol2 = Chem.MolFromSmiles(list_t[i])
        RDK_fps = [Chem.RDKFingerprint(x) for x in [mol1, mol2]]
        MACCS_fps = [MACCSkeys.GenMACCSKeys(x) for x in [mol1, mol2]]
#         AP_fps = [Pairs.GetAtomPairFingerprint(x) for x in [mol1, mol2]]
#         tts_fps = [Torsions.GetTopologicalTorsionFingerprintAsIntVect(x) for x in [mol1, mol2]]
        MG_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in [mol1, mol2]]
#         ECFP4_fps = [Chem.AllChem.GetMorganFingerprint(x,2) for x in [mol1, mol2]]
#         FCFP4_fps = [Chem.AllChem.GetMorganFingerprint(x,2, useFeatures=True) for x in [mol1, mol2]]
        RDKfp_sim.append(DataStructs.FingerprintSimilarity(RDK_fps[0], RDK_fps[1], metric=eval(metic_list[0])))
        MACCS_sim.append(DataStructs.FingerprintSimilarity(MACCS_fps[0], MACCS_fps[1], metric=eval(metic_list[1])))
#         AP_sim.append(DataStructs.FingerprintSimilarity(AP_fps[0], AP_fps[1], metric=eval(metic_list[0]))) 
#         tts_sim.append(DataStructs.FingerprintSimilarity(tts_fps[0], tts_fps[1], metric=eval(metic_list[0])))
        MGfp_sim.append(DataStructs.FingerprintSimilarity(MG_fps[0], MG_fps[1], metric=eval(metic_list[1])))
#         ECFP4_sim.append(DataStructs.FingerprintSimilarity(ECFP4_fps[0], ECFP4_fps[1], metric=eval(metic_list[1])))
#         FCFP4_sim.append(DataStructs.FingerprintSimilarity(FCFP4_fps[0], FCFP4_fps[1], metric=eval(metic_list[1])))
    return RDKfp_sim, MACCS_sim, MGfp_sim

def squeeze(lt):
    out=[]
    for i in lt:
        for j in i:
            out.append(j)
    return out

def to_index(i, n):
    return(i//n, i%n)

metic_list = ['DataStructs.TanimotoSimilarity', 'DataStructs.DiceSimilarity',
            'DataStructs.CosineSimilarity', 'DataStructs.SokalSimilarity',
            'DataStructs.RusselSimilarity', 'DataStructs.KulczynskiSimilarity',
             'DataStructs.McConnaugheySimilarity']