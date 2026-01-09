import os
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch, itertools

class MolData(InMemoryDataset):
    def __init__(self, root='./dataset', dataset=None, xd=None, y=None, transform=None, pre_transform=None, smile_graph=None):
        super(MolData, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data_list = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y, smile_graph)
            self.data_list = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['raw_file']

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, y, smile_graph):
        assert (len(xd) == len(y)), "smiles and labels must be the same length!"

        data = []
        for smiles in smile_graph.keys():
            if smiles is not None:
                leng, features, edge_index, edge_attr = smile_graph[smiles][0]
                for conditions, desc, relation, y_, cid, condition_array in smile_graph[smiles][1:]:
                    graph_data = DATA.Data(
                        x=torch.Tensor(features),
                        edge_index=torch.LongTensor(edge_index).transpose(1, 0).contiguous(),
                        edge_attr=torch.Tensor(edge_attr),
                        y=torch.FloatTensor([y_]),
                        smi=smiles,
                        cond=conditions.unsqueeze(0),
                        desc=desc.unsqueeze(0),
                        rela=relation,
                        cid=[cid],
                        condition_array=condition_array)
                    data.append(graph_data)

        print(len(data))
        torch.save(data, self.processed_paths[0])

    def get(self, idx):
        return self.data_list[idx]

    def len(self):
        return len(self.data_list)