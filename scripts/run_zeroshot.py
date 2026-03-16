import torch, argparse, os, sys
import torch.nn as nn
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from Dataset_test import MolData
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_ace import MolNet
from utils import get_logger, set_seed, metrics
from torch_geometric.data import DataLoader


def testing(model, test_loader, loss_f, metric, device, resu, task_i, prompt_i):
    loss_record, record_count = 0., 0.
    preds = torch.Tensor([])
    tars = torch.Tensor([])
    smiles_list = [] 
    
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            if data.y.size()[0] > 1:
                y = data.y.to(device)
                smiles_list.extend(data.smi)

                logits, relation, ms = model(data, task_i, prompt_i)

                loss = loss_f(logits.squeeze(), y.squeeze())
                loss = loss.mean()
                loss_record += float(loss.item())
                record_count += 1

                pred = logits.detach()
                preds = torch.cat([preds, pred.cpu()], 0)
                tars = torch.cat([tars, y.cpu()], 0)

        preds, tars = preds.squeeze().numpy(), tars.squeeze().numpy()
        mae = metric(preds, tars)

    epoch_loss = loss_record / record_count
    
    if resu:
        return epoch_loss, mae, preds, tars, smiles_list
    else:
        return epoch_loss, mae

def main(modelparm, dataset, pi, device, seed, batch_size, logger, met):
    logger.info(f'Running Test/Zero-shot on Dataset: {dataset}')

    if seed is not None:
        set_seed(seed)

    loss_f = nn.L1Loss(reduction='none').to(device)
    
    if met == 'mae':
        metric = metrics(mean_absolute_error)
    else:
        metric = metrics(root_mean_squared_error)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, 'dataset')
    
    testset = MolData(root=dataset_path, dataset=f'{dataset}_test')
    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    model = MolNet(modelparm).to(device)
    loss, test_mae, preds, tars, smiles = testing(model, test_loader, loss_f, metric, device, True, None, pi)
    
    logger.info(f"Test Finished. Loss: {loss:.4f}  MAE: {test_mae:.4f}")

    output_file = f'{dataset}_pred.csv'
    df = pd.DataFrame({
        'SMILES': smiles,
        'True_Value': tars,
        'Predicted_Value': preds
    })
    df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WodMol')
    parser.add_argument('--moldata', type=str, required=True, help='Dataset name (e.g., CHEMBL218)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--pretrain', type=str, default='checkpoints/model_CSLoss.pkl', help='Path to model weights')
    parser.add_argument('--pi', default=None, help='Task ID (int) for known tasks, None for zero-shot')
    parser.add_argument('--ft', default=False, type=str, help='Set True if you are using a fine-tuned checkpoint')
    parser.add_argument('--rela', type=str, default='False', help='Whether to use relation info (True/False)')
    parser.add_argument('--metric', type=str, choices=['rmse', 'mae'], default='mae', help='Metric')
    parser.add_argument('--seed', type=int, default=426, help="Random seed")
    parser.add_argument('--numtasks', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn_layers', type=int, default=4)
    parser.add_argument('--output_dim', type=int, default=512)
    parser.add_argument('--task_file', type=str, help="Task embedding file path")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    moldata = args.moldata
    input_dim = 40
    cond_dim = 512 
    cond_totlen = 1536 
    pi_lower = args.pi.lower().strip()
    if pi_lower in ['none', 'null', 'nan', '', 'None']:
        pi = None
    else:
        pi = int(args.pi)
    ft_lower = args.ft.lower().strip()
    if ft_lower in ['True', 'true']:
        ft = True
    elif ft_lower in ['False', 'false']:
        ft = False
    else:
        raise ValueError(f"Invalid value for --ft: {args.ft}")
    
    logf = f'log/Test_{moldata}.log'
    logger = get_logger(logf, moldata)
    
    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    modelparm = {
        'tasks': args.numtasks, 
        'output_dim': args.output_dim, 
        'attn_layers': args.attn_layers, 
        'cond_dim': cond_dim, 
        'cond_totlen': cond_totlen, 
        'dropout': args.dropout, 
        'input_dim': input_dim, 
        'rela': args.rela, 
        'device': args.device,
        'pretrain': args.pretrain,
        'task_embs': args.task_file,
        'pi': pi,
        'ft': ft
    }
    
    main(modelparm, moldata, pi, device, args.seed, args.batch_size, logger, args.metric)

