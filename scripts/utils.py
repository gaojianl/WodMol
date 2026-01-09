import torch, random
import torch.nn as nn
import numpy as np
import logging
from torch_geometric.data import DataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffoldSmiles(mol=mol, includeChirality=True)

def scaffold_split(smis, validrate=0.1, seed=0):
    scaffolds = {}
    for i, smi in enumerate(smis):
        scaffold = generate_scaffold(smi)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [i]
        else:
            scaffolds[scaffold].append(i)
    
    rng = np.random.RandomState(seed)
    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))
    valid_inds, train_inds = [], []
    n_total_valid = round(validrate * len(smis))
    for scaffold_set in scaffold_sets:
        if len(valid_inds) + len(scaffold_set) <= n_total_valid:
            valid_inds.extend(scaffold_set)
        else:
            train_inds.extend(scaffold_set)
    return train_inds, valid_inds


def load_data_singlevalid(train, test, batch_size, rate, seed, type_name):
    inhibition_indices = [i for i, d in enumerate(train) if d.condition_array[1] == type_name]
    non_inhibition_indices = [i for i, d in enumerate(train) if d.condition_array[1] != type_name]

    inhibition_smis = [train[i].smi for i in inhibition_indices]
    train_inds_inh, valid_inds_inh = scaffold_split(inhibition_smis, rate, seed)

    train_inds_inh = [inhibition_indices[i] for i in train_inds_inh]
    valid_inds_inh = [inhibition_indices[i] for i in valid_inds_inh]

    final_train_inds = non_inhibition_indices + train_inds_inh
    trainset = [train[i] for i in final_train_inds]
    validset = [train[i] for i in valid_inds_inh]

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

    return train_loader, valid_loader, test_loader


def load_data(train, test, batch_size, rate, seed):
    smis = [d.smi for d in train]
    train_inds, valid_inds = scaffold_split(smis, rate, seed)

    trainset = train[train_inds]
    validset = train[valid_inds]
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    return train_loader, valid_loader, test_loader


class metrics(nn.Module):
    def __init__(self, mae_f):
        super(metrics, self).__init__()
        self.mae_f = mae_f
    
    def forward(self, out, tar):
        if len(tar.shape) == 2:
            maes = []
            for i in range(tar.shape[1]):
                mae = self.mae_f(tar[:, i], out[:,i])
                maes.append(mae)
            mae = np.mean(maes)
        else:
            mae = self.mae_f(tar, out)
        return mae

def lr_lambda(epoch):
    if epoch < 10:
        return 1
    else:
        return 0.1 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class loss_rela(nn.Module):
    def __init__(self, met, device):
        super(loss_rela, self).__init__()
        if met == 'mae':
            self.loss = nn.L1Loss(reduction='none').to(device)
        else:
            self.loss = nn.MSELoss(reduction='none').to(device)

    def forward(self, pred, tar, rela):
        loss = self.loss(pred, tar)

        mask = torch.zeros_like(loss, dtype=torch.float32)
        mask[rela == 0] = 1  # =: 对应样本的损失不变
        mask[(rela == 1) & (pred > tar)] = 1  # <: pred > tar 时有效
        mask[(rela == 2) & (pred < tar)] = 1  # >: pred < tar 时有效

        loss = loss * mask
        return loss

def get_logger(filename, name=None, verbosity=1):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR}
    formatter = logging.Formatter("[%(asctime)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])#根据输入的verbosity参数（0到3），设置日志记录器的日志级别，控制记录输出的详细程度。
    fh = logging.FileHandler(filename, "a")#FileHandler，将日志信息写入指定的文件（filename）
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()#StreamHandler，将日志信息输出到控制台
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
