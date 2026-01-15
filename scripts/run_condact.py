import torch, argparse, sys, os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from Dataset_test import MolData
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_ace_noleakage import MolNet
from utils import get_logger, set_seed, loss_rela, load_data, metrics, lr_lambda


acelist = ['CHEMBL1871','CHEMBL218','CHEMBL244','CHEMBL236','CHEMBL234','CHEMBL219','CHEMBL238','CHEMBL4203','CHEMBL2047','CHEMBL4616','CHEMBL2034','CHEMBL262','CHEMBL231','CHEMBL264','CHEMBL2835','CHEMBL2971','CHEMBL237','CHEMBL233','CHEMBL4792','CHEMBL239','CHEMBL3979','CHEMBL235','CHEMBL4005','CHEMBL2147','CHEMBL214','CHEMBL228','CHEMBL287','CHEMBL204','CHEMBL1862']

def training(model, train_loader, optimizer, loss_f, metric, device, task_i, prompt_i, typ):
    loss_record, record_count = 0., 0.
    preds = torch.Tensor([]); tars = torch.Tensor([])
    model.train()

    for data in train_loader:
        if data.y.size()[0] > 1:
            y = data.y.to(device)
            
            logits, relation, ms = model(data, task_i, prompt_i)
            typs = torch.BoolTensor((np.array(ms) == typ)) 
            weights = torch.where(typs, torch.ones_like(typs), torch.ones_like(typs)*0.5).to(device)

            if relation is None:
                loss = loss_f(logits.squeeze(), y.squeeze())*weights
            else:
                loss = loss_f(logits.squeeze(), y.squeeze(), relation)
            loss = loss.mean()
            loss_record += float(loss.item())
            record_count += 1
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=2)
            optimizer.step()
            
            pred = logits.detach()
            preds = torch.cat([preds, pred.cpu()], 0); tars = torch.cat([tars, y.cpu()], 0)

    preds, tars = preds.squeeze().numpy(), tars.squeeze().numpy()
    mae = metric(preds, tars)
    
    epoch_loss = loss_record / record_count
    return epoch_loss, mae


def testing(model, test_loader, loss_f, metric, device, resu, task_i, prompt_i, typ):
    loss_record, record_count = 0., 0.
    preds = torch.Tensor([]); tars = torch.Tensor([])
    model.eval()
    with torch.no_grad():

        for data in test_loader:
            if data.y.size()[0] > 1:
                y = data.y.to(device)
                logits, relation, ms = model(data, task_i, prompt_i)
                typs = torch.BoolTensor((np.array(ms) == typ)) 
                weights = torch.where(typs, torch.ones_like(typs), torch.ones_like(typs)*0.5).to(device)

                if relation is None:
                    loss = loss_f(logits.squeeze(), y.squeeze())*weights
                else:
                    loss = loss_f(logits.squeeze(), y.squeeze(), relation)
                loss = loss.mean()
                loss_record += float(loss.item())
                record_count += 1

                pred = logits.detach()
                preds = torch.cat([preds, pred.cpu()], 0); tars = torch.cat([tars, y.cpu()], 0)

        preds, tars = preds.squeeze().numpy(), tars.squeeze().numpy()
        mae = metric(preds, tars)

    epoch_loss = loss_record / record_count
    if resu:
        return epoch_loss, mae, preds, tars
    else:
        return epoch_loss, mae



def main(modelparm, dataset, device, train_epoch, seed, fold, batch_size, rate, modelpath, logger, lr, met, savem):
    logger.info('Dataset: {}  train_epoch: {}'.format(dataset, train_epoch))
    task_i = acelist.index(dataset)
    prompt_i = None

    fold_result = []
    loss_f = nn.MSELoss(reduction='none').to(device)
    if rela:
        loss_f = loss_rela(met, device)
    if met == 'mae':
        metric = metrics(mean_absolute_error)
    else:
        metric = metrics(root_mean_squared_error)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, 'CondACT')
    
    trainset = MolData(root=dataset_path, dataset=f'{dataset}_train')
    testset = MolData(root=dataset_path, dataset=f'{dataset}_test')

    typ = testset[0].condition_array[1]
    
    fold_result = []
    for fol in range(1, fold+1):
        best_val_rmse, best_test_rmse = 9999., 9999.
        if seed is not None:
            seed_ = seed + fol-1
            set_seed(seed_)
        
        model = MolNet(modelparm).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logger.info('Dataset: {}  Fold: {:<4d}'.format(moldata, fol))

        train_loader, valid_loader, test_loader = load_data(trainset, testset, batch_size, rate[0], seed_)
        for i in range(train_epoch):
            loss, train_mae = training(model, train_loader, optimizer, loss_f, metric, device, task_i, prompt_i, typ)
            logger.info(f"Epoch {i + 1}/{train_epoch}  Train Loss: {loss:.4f} MAE: {train_mae:.4f}")
            
            if sche:
                scheduler.step()
            
            loss, valid_mae = testing(model, valid_loader, loss_f, metric, device, False, task_i, prompt_i, typ)
            logger.info(f"Epoch {i + 1}/{train_epoch}  Valid Loss: {loss:.4f} MAE: {valid_mae:.4f}")
            
            if savem:
                model_save_path = modelpath + '{}_{}_{}.pkl'.format(dataset, i, round(valid_mae,4))
                torch.save(model.state_dict(), model_save_path)
            
            if valid_mae < best_val_rmse:
                loss, test_mae = testing(model, test_loader, loss_f, metric, device, False, task_i, prompt_i, typ)
                logger.info(f"Epoch {i + 1}/{train_epoch}  Test  Loss: {loss:.4f} MAE: {test_mae:.4f}")
                best_val_rmse = valid_mae
                best_test_rmse = test_mae
        fold_result.append(best_test_rmse)
    res_mean, res_std = np.mean(fold_result, 0), np.std(fold_result, 0)
    logger.info(f"{dataset} {fold_result}")
    logger.info(f"{dataset} MEAN: {res_mean} STD: {res_std}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WodMol')
    parser.add_argument('--moldata', type=str, default='CHEMBL218', help='Dataset name')
    parser.add_argument('--numtasks', type=int, default=1, help='Number of tasks (default: 1).')
    parser.add_argument('--device', type=str, default='cuda:0', help='Which gpu to use if any (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training (default: 128)')
    parser.add_argument('--train_epoch', type=int, default=30, help='Number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--valrate', type=float, default=0.1, help='valid rate (default: 0.1)')
    parser.add_argument('--testrate', type=float, default=0., help='test rate (default: 0.)')
    parser.add_argument('--fold', type=int, default=5, help='Number of folds for cross validation (default: 5)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--attn_layers', type=int, default=4, help='Number of feature learning layers')
    parser.add_argument('--output_dim', type=int, default=512, help='Hidden size of embedding layer')
    parser.add_argument('--rela', type=bool, default=False, help='Whether to use activity raltionship (</>/=) when training')
    parser.add_argument('--seed', type=int, default=426, help = "Seed for splitting the dataset")
    parser.add_argument('--pretrain', type=str, default='checkpoints/model_noleakage.pkl', help = "Path of retrained weights")
    parser.add_argument('--task_file', type=str, default='CondACT/raw/ace_task.npy', help = "Path of the npy file containing task embedding") 
    parser.add_argument('--metric', type=str, choices=['rmse', 'mae'], default='mae', help='Metric to evaluate the regression performance')
    parser.add_argument('--savem', type=bool, default=False, help='Whether to save model checkpoints')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    moldata = args.moldata
    rate = [args.valrate, args.testrate]
    input_dim = 40
    cond_dim = 512 
    cond_totlen = 1536 
    numtasks = args.numtasks
    output_dim = args.output_dim
    rela = args.rela
    sche = True
    savem = args.savem
    
    logf = 'log/ACT_{}.log'.format(moldata)
    modelpath = 'log/checkpoint/'
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
    }
    
    fold_result = main(modelparm, moldata, device, args.train_epoch, args.seed, args.fold, args.batch_size, rate, modelpath, logger, args.lr, args.metric, savem)

    
