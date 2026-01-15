import networkx as nx
import numpy as np
import pandas as pd
import os, torch, argparse
import joblib, sys
from multiprocessing import Pool
from rdkit import Chem
from Dataset_test import MolData
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


des_list=[
    'This is the experimental condition —— BAO Label (The BioAssay Ontology classification): ',
    'This is the experimental condition —— Standard Type (The abbreviation of biological assay metrics reported): ',
    'This is the experimental condition —— Standard Unit (The unit of measurement for a biological experimental value): ',
    'This is the experimental condition —— Assay Organism: ',
    'This is the experimental condition —— Assay Tissue: ',
    'This is the experimental condition —— Smiles of non-covalent molecular assembly (co-crystals, solvates, salts, ion pairs, Hydrates): ', 
    'This is the experimental condition —— Incubation time: ', 
    'This is the experimental condition —— Temperature (℃): ',
    'This is the experimental condition —— pH: ',
    'This is the experimental condition —— Dosing/Compound concentration: ',
    'This is the experimental condition —— Assay method: ',
    'This is the experimental condition —— Route of administration: ',
    'This is the experimental condition —— Cell line (abbreviation): '
    ]

def get_relation(relation):
    if relation in ['<','<=']:
        return 1
    elif relation == '=':
        return 0
    elif relation in ['>','>=']:
        return 2
    else:
        return 0

def get_emb(sentence, tokenizer, model, device):
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    sentence_embedding = last_hidden_state.mean(dim=1)
    return sentence_embedding.detach().cpu()


# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(mol):
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(edges_list, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    return x, edge_index, edge_attr


cond2emb = {}
condmodel = joblib.load('dataset/raw/PCA1024model_cond.pkl')

def smiletopyg(smi, condition_array, desc, label, chemblid, tokenizer, model, device):
    g = nx.Graph()
    mol = Chem.MolFromSmiles(smi)
    c_size = mol.GetNumAtoms()
    
    features, edge_index, edge_attr = mol_to_graph_data_obj_simple(mol) 
    
    condition = []
    for cid, cond in enumerate(condition_array[:-1]):
        if cid != 7:
            if not pd.isna(cond):
                if cond not in cond2emb.keys():
                    emb = get_emb(des_list[cid]+cond, tokenizer, model, device)
                    emb = condmodel.transform(emb.numpy())[0]
                    cond2emb[cond] = emb
                else:
                    emb = cond2emb[cond]
            else:
                emb = np.zeros(1024)
        else:
            if not pd.isna(cond):
                if cond not in cond2emb.keys():
                    emb = get_emb(des_list[cid]+cond, tokenizer, model, device)
                    emb = condmodel.transform(emb.numpy())[0]
                    cond2emb[cond] = emb
                else:
                    emb = cond2emb[cond]
            else: 
                cond = '20 to 25 ℃'
                if cond not in cond2emb.keys():
                    emb = get_emb(des_list[cid]+cond, tokenizer, model, device)
                    emb = condmodel.transform(emb.numpy())[0]
                    cond2emb[cond] = emb
                else:
                    emb = cond2emb[cond]
        condition.append(emb)
    condition = np.array(condition)
    
    g = [[c_size, features, edge_index, edge_attr], [torch.FloatTensor(condition), torch.FloatTensor(desc), get_relation(condition_array[-1]), label, chemblid, condition_array]] 
    return [smi, g] 


def get_absolute_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, relative_path))

def process_dataset(moldata_name, data_type, llm_path, device, numtasks, label_col, tokenizer, model, descmodel):
    dataset_root = get_absolute_path('../dataset')
    processed_dir = os.path.join(dataset_root, 'processed')
    raw_dir = os.path.join(dataset_root, 'raw')

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    processed_data_file = os.path.join(processed_dir, f'{moldata_name}_{data_type}.pt')
    if os.path.isfile(processed_data_file):
        print(f"Processed file already exists: {processed_data_file}")
        return False

    raw_data_file = os.path.join(raw_dir, f'{moldata_name}_{data_type}_llamacond.csv')
    if not os.path.isfile(raw_data_file):
        raise FileNotFoundError(f"Raw CSV file not found: {raw_data_file}")

    df = pd.read_csv(raw_data_file)
    df = df.dropna(subset=['Smiles', 'standard_type', 'standard_value'])

    compound_iso_smiles = np.array(df['Smiles'])
    
    if numtasks > 1:
        label_columns = label_col.split(',')
        values = np.array(df[label_columns])
    else:
        values = np.array(df['standard_value'])

    condition_columns = [
        'bao_label', 'standard_type', 'standard_units', 'assay_organism', 
        'assay_tissue', 'Solvents', 'Incubation Time', 'Temperature', 'pH', 
        'Compound Concentration', 'Assay Method', 'Compound Administration Method', 
        'Cell Line', 'standard_relation'
    ]
    conditions = np.array(df[condition_columns].astype(str))
    cids = df['assay_chembl_id'].values.tolist()
    descs = df['assay_description'].values.tolist()
    
    desc_uniq = list(set(descs))
    emb_uniq = []
    desc2emb = {}
    
    for desc in desc_uniq:
        emb = get_emb(desc, tokenizer, model, device)
        emb_uniq.append(emb.squeeze(0).numpy())
    
    emb_uniq = descmodel.transform(np.array(emb_uniq))
    
    for d, e in zip(desc_uniq, emb_uniq):
        desc2emb[d] = e
    
    desc_embs = [desc2emb[desc] for desc in descs]

    smile_graph = {}
    smis = []
    y = []
    
    for i, (smi, label, condition_array, desc, cid, desct) in enumerate(zip(
        compound_iso_smiles, values, conditions, desc_embs, cids, descs)):
        
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            smi_canonical = Chem.MolToSmiles(mol)
            y.append(label)
            smis.append(smi_canonical)
            
            smi_processed, g = smiletopyg(smi_canonical, condition_array, desc, label, cid, tokenizer, model, device)
            
            if smi_processed in smile_graph:
                smile_graph[smi_processed].append(g[1])
            else:
                smile_graph[smi_processed] = [g[0], g[1]]
        
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(compound_iso_smiles)} molecules")

    print(f"  Saving to {processed_data_file}...")
    MolData(root=dataset_root, dataset=f'{moldata_name}_{data_type}', xd=smis, y=y, smile_graph=smile_graph)
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WodMol Preprocessing')
    parser.add_argument('--moldata', type=str, required=True, help='Dataset name (e.g., CHEMBL218)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--numtasks', type=int, default=1, help='Number of tasks')
    parser.add_argument('--testing', type=str, default=False, help='Only process the test dataset.')
    parser.add_argument('--label_col', type=str, help='Name of label column(s). For single task, use standard_value; For multiple tasks, use commas to separate multiple column names.')
    parser.add_argument('--llm_path', type=str, required=True, help='Path of the llama model')
    args = parser.parse_args()

    if args.llm_path is None:
        raise ValueError("llm_path is required")

    dataset_root = get_absolute_path('../dataset')
    pca_model_path = os.path.join(dataset_root, 'raw', 'PCA1024model_desc.pkl')

    model_id = args.llm_path
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(args.device)
    descmodel = joblib.load(pca_model_path)
    
    process_dataset(
        moldata_name=args.moldata,
        data_type='test',
        llm_path=args.llm_path,
        device=args.device,
        numtasks=args.numtasks,
        label_col=args.label_col,
        tokenizer=tokenizer,
        model=model,
        descmodel=descmodel
    )
    
    if args.testing in ['False', 'false']:
        process_dataset(
            moldata_name=args.moldata,
            data_type='train',
            llm_path=args.llm_path,
            device=args.device,
            numtasks=args.numtasks,
            label_col=args.label_col,
            tokenizer=tokenizer,
            model=model,
            descmodel=descmodel
        )

