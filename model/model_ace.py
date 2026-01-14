import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
import numpy as np
from model.cmoe import MoE_cond
from model.util import FiLM_layer, AttPool, CondTrans
from model.gnns import CABgnn, SPBgnn


class MolNet(nn.Module):
    def __init__(self, modelparm):
        super(MolNet, self).__init__()
        tasks, output_dim, attn_layers, cond_dim, cond_totlen, dropout, input_dim, rela, device = modelparm['tasks'], modelparm['output_dim'], modelparm['attn_layers'], modelparm['cond_dim'], modelparm['cond_totlen'], modelparm['dropout'], modelparm['input_dim'], modelparm['rela'], modelparm['device']
        task_embs_file = modelparm['task_embs']
        pi = modelparm['pi']
        ft = modelparm['ft']
        self.device = device
        self.rela = rela
        self.task_embs = torch.tensor(np.load(task_embs_file), dtype=torch.float32)
        if pi is not None:
            self.virtual_node_embedding = nn.Parameter(torch.randn(19776, output_dim))
        else:
            self.virtual_node_embedding = nn.Parameter(torch.zeros(1, output_dim))
        self.tlin = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(1024, output_dim))

        self.emb = CABgnn(output_dim, attn_layers, dropout, input_dim, device) 
        self.emb_org = SPBgnn(output_dim, attn_layers, dropout, input_dim, device) 

        self.films = nn.ModuleList([FiLM_layer(output_dim, cond_totlen, dropout) for i in range(int(attn_layers/2))])
        self.Ctrans = CondTrans(cond_dim)
        self.readout = AttPool(output_dim*2)
        self.dp = nn.Dropout(dropout)
        self.moe = MoE_cond(output_dim*2, expert_dim=128, num_experts=7, k=2, num_generalists=1, cond_totlen=cond_totlen)
        self.pff = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, tasks))
        self.reset_params()
        if modelparm['pretrain'] is not None:
            print('loading model ...')
            state_dict = torch.load(modelparm['pretrain'], weights_only=True, map_location='cuda:0')
            model_dict = self.state_dict()
            if pi is not None:
                modules_to_load = ['emb', 'emb_org', 'tlin', 'films', 'Ctrans', 'readout', 'moe', 'pff', 'virtual_node_embedding']
            else:
                if ft:
                    modules_to_load = ['emb', 'emb_org', 'tlin', 'films', 'Ctrans', 'readout', 'moe', 'pff', 'virtual_node_embedding']
                else:
                    modules_to_load = ['emb', 'emb_org', 'tlin', 'films', 'Ctrans', 'readout', 'moe', 'pff']
            filtered_state_dict = {k: v for k, v in state_dict.items() if any(k.startswith(module) for module in modules_to_load)}
            model_dict.update(filtered_state_dict)
            self.load_state_dict(model_dict)

    def reset_params(self):
        for weight in self.parameters():
            if len(weight.size()) > 1:
                nn.init.xavier_normal_(weight)
    
    def forward(self, data, ti, pi):
        x, edge_index = data.x.to(self.device).int(), data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device)
        batch = data.batch.to(self.device)
        ms = []
        for c in data.condition_array:
            ms.append(c[1])
        cond = data.cond.to(self.device)
        desc = data.desc.to(self.device)
        cond = torch.cat([cond, desc.unsqueeze(1)], 1)
        batch_size = cond.size()[0]
        _, cond_flat = self.Ctrans(cond)
        
        if ti is not None:
            task_embs = self.task_embs[ti].unsqueeze(0).repeat(batch_size, 1).to(self.device)
        else:
            task_embs = self.task_embs.repeat(batch_size, 1).to(self.device)
        task_embs = self.tlin(task_embs)
        
        if pi is None:
            vne = self.virtual_node_embedding.repeat(batch_size, 1)
        else:
            vne = self.virtual_node_embedding[pi].unsqueeze(0).repeat(batch_size, 1)

        # DoubGNN2
        x_new = self.emb(x, edge_index, edge_attr, batch, cond_flat, self.films, task_embs) 
        x_org = self.emb_org(x, edge_index, edge_attr, batch, vne) 

        x = torch.cat([x_org, x_new], -1)
        x_batch, _ = to_dense_batch(x, batch)
        x_p = self.readout(x_batch)
        x_p = self.moe(self.dp(x_p), cond_flat)
        logits = self.pff(x_p)

        if self.rela:
            rela = data.rela.to(self.device)
        else:
            rela = None
        return logits.squeeze(-1), rela, ms
