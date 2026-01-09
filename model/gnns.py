import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from model.gin import GINConv

num_atom_type = 120   #including the extra mask tokens
num_chirality_tag = 3
num_bond_type = 6     #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class CABgnn(nn.Module):
    def __init__(self, output_dim=128, attn_layers=4, dropout=0.1, input_dim=40, device='cuda:0'):
        super(CABgnn, self).__init__()
        self.device = device 
        self.layer_num = attn_layers 
        self.num_layer = attn_layers
        self.drop_ratio = dropout
        self.outputdim = output_dim
        self.x_embedding1 = nn.Embedding(num_atom_type, output_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, output_dim)
        self.gnns = torch.nn.ModuleList()
        for _ in range(attn_layers):
            self.gnns.append(GINConv(output_dim, aggr="add"))
        self.nms = torch.nn.ModuleList()
        for _ in range(attn_layers):
            self.nms.append(torch.nn.BatchNorm1d(output_dim))

    def add_virtual_nodes_batch_vectorized(self, x, edge_index, edge_attr, batch, task_embs):
        num_nodes = x.size(0)
        num_graphs = batch.max().item() + 1
        
        virtual_node_embeds = task_embs
        
        virtual_node_indices = num_nodes + torch.arange(num_graphs, device=x.device)
        
        x_new = torch.cat([x, virtual_node_embeds], dim=0)
        batch_new = torch.cat([batch, torch.arange(num_graphs, device=x.device)], dim=0)
        
        node_counts = torch.bincount(batch, minlength=num_graphs)

        src = virtual_node_indices.repeat_interleave(node_counts)
        dst = torch.arange(num_nodes, device=x.device)
        
        virtual_edge_index = torch.stack([torch.cat([src, dst]),torch.cat([dst, src])], dim=0)
        
        edge_index_new = torch.cat([edge_index, virtual_edge_index], dim=1)
        edge_attr_new = torch.cat([edge_attr, torch.tensor([[5, 0]]).repeat(virtual_edge_index.size(1), 1).to(self.device)], dim=0)
        
        return x_new, edge_index_new, edge_attr_new, batch_new

    def forward(self, x, edge_index, edge_attr, batch, task_embs, films, teb):
        g = batch.max().item()+1
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        x, edge_index, edge_attr, _ = self.add_virtual_nodes_batch_vectorized(x, edge_index, edge_attr, batch, teb)

        for layer in range(self.num_layer):
            x = self.gnns[layer](x, edge_index, edge_attr)
            
            if layer in [1, 3]:
                xv = x[-g:, :]
                x_batch, mask = to_dense_batch(x[:-g,:], batch)
                fl = layer // 3 
                x_batch = films[fl](x_batch, task_embs)
                x = torch.masked_select(x_batch, mask.unsqueeze(-1))
                x = x.reshape(-1, self.outputdim)
                x = torch.cat([x, xv], 0)

            x = self.nms[layer](x)
            if layer == self.num_layer - 1:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

        return x[:-g,:]

class SPBgnn(nn.Module):
    def __init__(self, output_dim=128, attn_layers=4, dropout=0.1, input_dim=40, device='cuda:0'):
        super(SPBgnn, self).__init__()
        self.device = device 
        self.layer_num = attn_layers 
        self.num_layer = attn_layers
        self.drop_ratio = dropout
        self.outputdim = output_dim
        self.x_embedding1 = nn.Embedding(num_atom_type, output_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, output_dim)
        self.gnns = torch.nn.ModuleList()
        for _ in range(attn_layers):
            self.gnns.append(GINConv(output_dim, aggr="add"))
        self.nms = torch.nn.ModuleList()
        for _ in range(attn_layers):
            self.nms.append(torch.nn.BatchNorm1d(output_dim))
        
    def add_virtual_nodes_batch_vectorized(self, x, edge_index, edge_attr, batch, task_embs):
        num_nodes = x.size(0)
        num_graphs = batch.max().item() + 1
        
        virtual_node_embeds = task_embs
        
        virtual_node_indices = num_nodes + torch.arange(num_graphs, device=x.device)
        
        x_new = torch.cat([x, virtual_node_embeds], dim=0)
        batch_new = torch.cat([batch, torch.arange(num_graphs, device=x.device)], dim=0)
        
        node_counts = torch.bincount(batch, minlength=num_graphs)

        src = virtual_node_indices.repeat_interleave(node_counts)
        dst = torch.arange(num_nodes, device=x.device)
        
        virtual_edge_index = torch.stack([torch.cat([src, dst]),torch.cat([dst, src])], dim=0)
        
        edge_index_new = torch.cat([edge_index, virtual_edge_index], dim=1)
        edge_attr_new = torch.cat([edge_attr, torch.tensor([[5, 0]]).repeat(virtual_edge_index.size(1), 1).to(self.device)], dim=0)
        
        return x_new, edge_index_new, edge_attr_new, batch_new

    def forward(self, x, edge_index, edge_attr, batch, task_embs):
        g = batch.max().item()+1
        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        x, edge_index, edge_attr, _ = self.add_virtual_nodes_batch_vectorized(x, edge_index, edge_attr, batch, task_embs)

        for layer in range(self.num_layer):
            x = self.gnns[layer](x, edge_index, edge_attr)
            x = self.nms[layer](x)
            if layer == self.num_layer - 1:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

        return x[:-g,:]
