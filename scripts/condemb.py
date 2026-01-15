import transformers
import torch, os, argparse
import pandas as pd
from tqdm import tqdm
import logging
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
remover = SaltRemover()
from rdkit.Chem import rdmolops
logging.getLogger("transformers").setLevel(logging.ERROR)  # 屏蔽所有 transformers 警告


prompt_am = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with extracting specific Assay Method from a given experimental protocol or description. Identify only specific proprietary experimental techniques or procedures such as "HTRF", "ELISA", without including common phrases that describe general experimental setups or contexts. Only extract the names of the experimental techniques that are widely recognized and have proprietary or commonly accepted abbreviations. If none are found, return "None". Extract exactly as the original information, do not speculate or imply. <|eot_id|>
<|start_header_id|>user<|end_header_id|>
HTRF Biochemical Assay: 250 nL of diluted compounds in DMSO were dispensed into the wells of a dry 384-well Black plate using a Labcyte Echo 555 acoustic dispenser. <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Assay Method = HTRF Biochemical Assay
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibition of 5-lipoxygenase in human neutrophil assessed as remaining activity at 10 uM preincubated for 15 mins before substrate arachidonic acid addition measured after 10 mins by HPLC method.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Assay Method = HPLC
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibitory activity against 5-lipoxygenase.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Assay Method = None
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''
prompt_it = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with extracting specific Incubation Time from a given experimental protocol or description. If multiple incubation times are mentioned (e.g., pre-incubation and reaction incubation), include the most important one for the assay. Use 'minutes' as units. 1 hour = 60 minutes. Do not extract preincubation time, return "None" if only preincubation time is mentioned. Extract exactly as the original information, do not speculate or imply. <|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibition of human carbonic anhydrase-2 incubated for 15 mins by stopped-flow CO2 hydration assay<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Incubation Time = 15 minutes
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibition of 5-lipoxygenase in human neutrophil assessed as remaining activity at 10 uM preincubated for 15 mins before substrate arachidonic acid addition measured after 10 mins by HPLC method.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Incubation Time = 10 minutes
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibitory activity against 5-lipoxygenase.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Incubation Time = None
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''


prompt_t_ph = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with extracting specific pH value from a given experimental protocol or description. Extract the pH and temperature value if provided. Extract exactly as the original information, do not speculate or imply. <|eot_id|>
<|start_header_id|>user<|end_header_id|>
Binding affinity to HIV1 protease V32I/I47V/V82I triple mutant at pH 5.8 and 28 degC by ITC method. <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Temperature = 28 ℃
pH = 5.8
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibition of 5-lipoxygenase in human neutrophil assessed as remaining activity at 10 uM preincubated for 15 mins before substrate arachidonic acid addition measured after 10 mins by HPLC method. <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Temperature = None
pH = None
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
HotSpot Technology Assay: The DYRK1A and DYRK1B kinase assays to determine IC50 values were performed by Reaction Biology Corporation using HotSpot technology Worldwide website: reactionbiology.com, Malvern, PA). Kinase reaction with specific kinase/ substrate pair along with required cofactors was carried out in 20 mM Hepes pH 7.5, 10 mM MgCl2, 1 mM EGTA, 0.02% Brij35, 0.02 mg/ml BSA, 0.1 mM Na3VO4, 2 mM DTT, 1% DMSO.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Temperature = None
pH = 7.5
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''

prompt_cam = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with extracting specific compound administration method from a given experimental protocol or description.  Identify how the compounds were administered to a biological subject (if there is compound administration method detected, return only abbreviation format. where i.v. means Intravenous, p.o. means oral or Per os, s.c. means Subcutaneous. Extract exactly as the original information, do not speculate or imply.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibition of 5-lipoxygenase in human neutrophil assessed as remaining activity at 10 uM preincubated for 15 mins before substrate arachidonic acid addition measured after 10 mins by HPLC method.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Compound Administration Method = None
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
The ability of each test compound to inhibit albino rabbit iris and ciliary body carbonic anhydrase post topical instillation was assessed ex vivo.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Compound Administration Method = topical instillation
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Dose required to inhibit LPS (20 mg/kg injected i.v. 1 hr before dosing) induced TNF-alpha release in mice upon oral administration.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Compound Administration Method = i.v.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''

prompt_cc = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with extracting specific compound concentration from a given experimental protocol or description. Extract the concentration of the administered compound. Do not extract the concentration of the substrate, enzyme or other components that are not the tested compound in the assay. If only a volume is mentioned but no concentration is provided, leave this field as "None". 10'x means 10^x, not 10^(-x). Extract exactly as the original information, do not speculate or imply. <|eot_id|>
<|start_header_id|>user<|end_header_id|>
Binding affinity to HIV1 protease V32I/I47V/V82I triple mutant at pH 5.8 and 28 degC by ITC method.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Compound Concentration = None
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibition of 5-lipoxygenase in human neutrophil assessed as remaining activity at 10 uM preincubated for 15 mins before substrate arachidonic acid addition measured after 10 mins by HPLC method.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Compound Concentration = 10 uM
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibitory activity against 5-lipoxygenase.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Compound Concentration = None
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''

prompt_cl = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with extracting specific cell type from a given experimental protocol or description. Extract the cell type used in the experiment if provided and do not include species information. Statement like 'A is expressed in B' normally means B is the cell type that we should extract. Statement like 'Inhibition of A' normally means A is the tested protein. If only A is mentioned, return None. For fields where no information is provided in the text, return "None". Extract exactly as the original information, do not speculate or imply.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Binding affinity to HIV1 protease V32I/I47V/V82I triple mutant at pH 5.8 and 28 degC by ITC method.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Cell Line = None
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibition of 5-lipoxygenase in human neutrophil assessed as remaining activity at 10 uM preincubated for 15 mins before substrate arachidonic acid addition measured after 10 mins by HPLC method.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Cell Line = human neutrophil
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Inhibitory activity against 5-lipoxygenase.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Cell Line = None
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''


def get_cond(file, pipeline):
    output_name = file.split('.')[0] + '_llamacond.csv'
    if not os.path.exists(output_name):
        df=pd.read_csv(file)
        results = df['canonical_smiles'].apply(presmi)

        df['Smiles'] = results.apply(lambda x: x[0] if x is not None else None)
        df['Solvents'] = results.apply(lambda x: ' '.join(x[1:]) if x is not None and len(x) > 1 else None)
        df = df[df['Smiles'].notnull()]
        prompt_dict = {prompt_am:'Assay Method',prompt_it:'Incubation Time',prompt_t_ph:'Temperature, pH',prompt_cam:'Compound Administration Method',prompt_cc:'Compound Concentration',prompt_cl:'Cell Line'}
    
        for i in tqdm(range(len(df))):
            result_dict = {}
            for prompt in prompt_dict.keys():
                d_prompt={}
                if prompt_dict[prompt] == 'Cell Line':
                    assay_cell_type = df.iloc[i]['assay_cell_type']
                    if pd.isnull(assay_cell_type) == False:
                        d_prompt['Cell Line'] = assay_cell_type
                        result_dict.update(d_prompt)
                        continue
                if prompt_dict[prompt] == 'Compound Administration Method':
                    BAO_Format = df.iloc[i]['bao_label']
                    if BAO_Format != 'organism-based format':
                        d_prompt['Compound Administration Method'] = None
                        result_dict.update(d_prompt)
                        continue
                input = df.iloc[i]['assay_description']
                
                input = prompt + input + "<|eot_id|> " + "\n<|start_header_id|>assistant<|end_header_id|>"
                
                output = pipeline(input, max_new_tokens=500, return_full_text=False)[0]['generated_text']
                torch.cuda.empty_cache()
                
                # 逐行解析
                output_ = output.strip().split("\n")
                for line in output_:
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()  # 去除键两边的空格
                        value = value.strip()  # 去除值两边的空格
    
                        # 将 "None" 转换为 Python 的 None
                        if value == "None":
                            d_prompt[key] = None
                        else:
                            d_prompt[key] = value
                if d_prompt=={}:
                    if prompt == prompt_t_ph:
                        d_prompt['Temperature'] = None
                        d_prompt['pH'] = None
                    else:
                        d_prompt[prompt_dict[prompt]] = None               
                result_dict.update(d_prompt)

            result_df_ = pd.DataFrame([result_dict])
            # 打印结果
            if i == 0:
                result_df = result_df_
            else:
                result_df = pd.concat([result_df, result_df_], axis=0)
        df = df.reset_index(drop=True)
        result_df = result_df.reset_index(drop=True)
        df_all = pd.concat([df, result_df], axis=1)
        df_all.to_csv(output_name, index=False)

def presmi(smi):
    record = True
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        record = False
    if mol is not None:
        fragments = rdmolops.GetMolFrags(mol, asMols=True)
        frag_smis = [Chem.MolToSmiles(frag) for frag in fragments]
        frag_smis.sort(key=len, reverse=True)
        smi = frag_smis[0]
        mol = Chem.MolFromSmiles(smi)
    else:
        record = False
    
    if record:
        return frag_smis
    else:
        return None 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WodMol')
    parser.add_argument('--moldata', type=str, required=True, help='Dataset name (e.g., CHEMBL218)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--testing', type=str, default=False, help='True: Only process the test dataset.')
    parser.add_argument('--llm_path', type=str, help="Path of the llama file (/t9k/mnt/.cache/kagglehub/models/metaresearch/llama-3.1/transformers/8b-instruct/1)")
    args = parser.parse_args()

    if args.llm_path is None:
        raise ValueError("llm_path is required when task_desc is not provided")
    model_id = args.llm_path
    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map=args.device)

    if args.testing in ['False', 'false']:
        get_cond('dataset/raw/'+args.moldata+'_train.csv', pipeline)
    get_cond('dataset/raw/'+args.moldata+'_test.csv', pipeline)

