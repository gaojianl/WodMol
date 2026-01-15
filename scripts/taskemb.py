import transformers, joblib
import torch, argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel

prompt_inh = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with providing a task description (a paragraph including definition, which pathway it belongs to (if any), function, function of its inhibitors/Antagonists/blockers(must include)) for a given bioassay target. This paragraph should be concise and without nonsense; its content should be supported by established knowledge. If the name contains different proteins or protein-protein interaction. You should concise these proteins into one paragraph and just give description of their relation and functions of their interaction instead of introducing the protein separately<|eot_id|>
<|start_header_id|>user<|end_header_id|>
AKT <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
AKT, or Protein Kinase B (PKB), is a serine/threonine kinase that is a key component of the PI3K/AKT/mTOR signaling pathway, which regulates critical cellular processes such as growth, survival, metabolism, and proliferation. Activated by phosphorylation through upstream signals like PI3K and PDK1, AKT phosphorylates various substrates to inhibit apoptosis (e.g., by inactivating BAD), promote glucose uptake (e.g., via GLUT4 translocation), enhance protein synthesis (e.g., through mTOR activation), and regulate the cell cycle. Inhibitors of AKT block its activation or kinase activity, making them significant in cancer therapy for inducing apoptosis and slowing tumor progression, as well as in metabolic disorders for modulating insulin signaling and glucose metabolism.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Leukocyte elastase.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Leukocyte elastase, also known as neutrophil elastase (NE), is a serine protease predominantly found in neutrophils, where it plays a vital role in the innate immune response. It is involved in breaking down extracellular matrix components, such as elastin, collagen, and proteoglycans, during inflammation and tissue remodeling. Leukocyte elastase is part of the serine protease family and contributes to the clearance of pathogens but can also cause tissue damage if its activity is unregulated. Overactivity of leukocyte elastase has been implicated in various diseases, including chronic obstructive pulmonary disease (COPD), cystic fibrosis, and acute lung injury. Inhibitors of leukocyte elastase, such as alpha-1 antitrypsin or synthetic elastase inhibitors, function to neutralize its proteolytic activity, thereby reducing inflammation and preventing tissue destruction in inflammatory and degenerative diseases.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''

prompt_ago = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with providing a task description (a paragraph including definition, which pathway it belongs to (if any), function, function of its agonists(must include)) for a given bioassay target. This paragraph should be concise and without nonsense; its content should be supported by established knowledge. If the name contains different proteins or protein-protein interaction. You should concise these proteins into one paragraph and just give description of their relation and functions of their interaction instead of introducing the protein separately<|eot_id|>
<|start_header_id|>user<|end_header_id|>
GLP-1R <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
GLP-1R, or Glucagon-like Peptide-1 Receptor, is a G-protein-coupled receptor (GPCR) that is a critical component of the incretin signaling pathway, which regulates glucose metabolism. Activated by the hormone GLP-1, GLP-1R stimulates insulin secretion from pancreatic beta cells, inhibits glucagon release, slows gastric emptying, and promotes satiety, contributing to glucose homeostasis and weight regulation. Agonists of GLP-1R, such as GLP-1 receptor agonists, mimic the effects of GLP-1 and are widely used in the treatment of type 2 diabetes and obesity. These agonists improve blood glucose control, support weight loss, and may provide cardiovascular benefits.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Melanocortin receptor 4
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Melanocortin receptor 4 (MC4R) is a G-protein-coupled receptor (GPCR) predominantly expressed in the central nervous system and is a key regulator of energy homeostasis and appetite. It is part of the melanocortin signaling pathway and is activated by melanocortin peptides, such as α-melanocyte-stimulating hormone (α-MSH). MC4R signaling suppresses appetite and increases energy expenditure, playing a crucial role in maintaining body weight. Agonists of MC4R enhance these effects and are being explored as potential treatments for obesity and related metabolic disorders, while mutations in MC4R are associated with monogenic obesity.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''

prompt_deg = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with providing a task description (a paragraph including definition, which pathway it belongs to (if any), function, function of its degraders(must include)) for a given bioassay target. This paragraph should be concise and without nonsense; its content should be supported by established knowledge. If the name contains different proteins or protein-protein interaction. You should concise these proteins into one paragraph and just give description of their relation and functions of their interaction instead of introducing the protein separately<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Estrogen receptor alpha <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Estrogen receptor alpha (ERα) is a nuclear hormone receptor that functions as a ligand-activated transcription factor in the estrogen signaling pathway. It is primarily activated by the hormone estrogen and plays a central role in regulating gene expression involved in reproductive development, bone health, cardiovascular function, and cell proliferation. ERα is highly expressed in estrogen-sensitive tissues such as the breast and uterus. Dysregulation or overexpression of ERα is implicated in the development and progression of hormone-dependent cancers, particularly breast cancer. Degraders of ERα, such as selective estrogen receptor degraders (SERDs), bind to the receptor, inducing a conformational change that leads to its ubiquitination and proteasomal degradation. These degraders are used therapeutically to inhibit ERα-driven signaling and combat estrogen receptor-positive breast cancers, especially those resistant to standard therapies.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Cereblon/BCR/ABL
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Cereblon (CRBN) and BCR/ABL interact within different biological contexts, but both are involved in the pathogenesis of hematologic malignancies. CRBN is a key component of an E3 ubiquitin ligase complex that mediates targeted protein degradation, and its function is modulated by immunomodulatory drugs (IMiDs) like thalidomide and lenalidomide. These drugs alter CRBN’s substrate specificity, leading to the degradation of transcription factors such as Ikaros (IKZF1) and Aiolos (IKZF3), which is pivotal in treating multiple myeloma. BCR/ABL, a fusion oncoprotein created by the Philadelphia chromosome translocation, is a constitutively active tyrosine kinase that drives the proliferation of chronic myeloid leukemia (CML) and some forms of acute lymphoblastic leukemia (ALL). Targeting BCR/ABL with tyrosine kinase inhibitors (TKIs) is a standard treatment for these leukemias. Although CRBN and BCR/ABL operate through different mechanisms, combining CRBN modulators with BCR/ABL inhibitors could enhance therapeutic efficacy, particularly in resistant leukemias.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''

prompt_mod = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with providing a task description (a paragraph including definition, which pathway it belongs to (if any), function, function of its modulators(must include)) for a given bioassay target. This paragraph should be concise and without nonsense; its content should be supported by established knowledge. If the name contains different proteins or protein-protein interaction. You should concise these proteins into one paragraph and just give description of their relation and functions of their interaction instead of introducing the protein separately<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Integrin alpha-V/beta-1 <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Integrin alpha-V/beta-1 (αVβ1) is a heterodimeric transmembrane receptor belonging to the integrin family, which mediates cell adhesion and signaling by interacting with extracellular matrix (ECM) proteins, such as fibronectin and vitronectin. This integrin plays a critical role in cellular processes such as migration, proliferation, and survival, particularly in development, wound healing, and angiogenesis. αVβ1 is involved in pathways like focal adhesion kinase (FAK) and PI3K/AKT signaling, linking ECM interactions to intracellular responses. Modulators of αVβ1, including inhibitors and allosteric regulators, can disrupt its binding to ECM ligands, thereby inhibiting angiogenesis and tumor progression. These modulators are under investigation for therapeutic applications in cancer, fibrosis, and other integrin-associated diseases.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Muscarinic acetylcholine receptor M1
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Muscarinic acetylcholine receptor M1 (M1 mAChR) is a G-protein-coupled receptor (GPCR) primarily expressed in the central nervous system, particularly in the brain's cortex and hippocampus. It is involved in the cholinergic signaling pathway, where its activation by acetylcholine regulates cognitive functions such as learning, memory, and attention, as well as neuronal excitability. M1 mAChR also mediates various physiological processes, including smooth muscle contraction and glandular secretion. Modulators of M1 mAChR, such as agonists and positive allosteric modulators, enhance acetylcholine signaling, showing potential for the treatment of cognitive disorders like Alzheimer's disease and schizophrenia. Antagonists, conversely, may be used to manage conditions like overactive bladder or motion sickness by inhibiting excessive cholinergic activity.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''


prompt_ai = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with providing a task description (a paragraph including definition, which pathway it belongs to (if any), function, function of its allosteric inhibitors(must include)) for a given bioassay target. This paragraph should be concise and without nonsense; its content should be supported by established knowledge. If the name contains different proteins or protein-protein interaction. You should concise these proteins into one paragraph and just give description of their relation and functions of their interaction instead of introducing the protein separately<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Integrin alpha-V/beta-1 <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Integrin alpha-V/beta-1 (αVβ1) is a heterodimeric transmembrane receptor belonging to the integrin family, which mediates cell adhesion and signaling by interacting with extracellular matrix (ECM) proteins, such as fibronectin and vitronectin. This integrin plays a critical role in cellular processes such as migration, proliferation, and survival, particularly in development, wound healing, and angiogenesis. αVβ1 is involved in pathways like focal adhesion kinase (FAK) and PI3K/AKT signaling, linking ECM interactions to intracellular responses. Modulators of αVβ1, including inhibitors and allosteric regulators, can disrupt its binding to ECM ligands, thereby inhibiting angiogenesis and tumor progression. These modulators are under investigation for therapeutic applications in cancer, fibrosis, and other integrin-associated diseases.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Muscarinic acetylcholine receptor M1
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Muscarinic acetylcholine receptor M1 (M1 mAChR) is a G-protein-coupled receptor (GPCR) primarily expressed in the central nervous system, particularly in the brain's cortex and hippocampus. It is involved in the cholinergic signaling pathway, where its activation by acetylcholine regulates cognitive functions such as learning, memory, and attention, as well as neuronal excitability. M1 mAChR also mediates various physiological processes, including smooth muscle contraction and glandular secretion. Modulators of M1 mAChR, such as agonists and positive allosteric modulators, enhance acetylcholine signaling, showing potential for the treatment of cognitive disorders like Alzheimer's disease and schizophrenia. Antagonists, conversely, may be used to manage conditions like overactive bladder or motion sickness by inhibiting excessive cholinergic activity.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''

prompt_mi = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with providing a task description (a paragraph including definition, which pathway it belongs to (if any), function, function of its allosteric modulators(must include)) for a given bioassay target. This paragraph should be concise and without nonsense; its content should be supported by established knowledge. If the name contains different proteins or protein-protein interaction. You should concise these proteins into one paragraph and just give description of their relation and functions of their interaction instead of introducing the protein separately<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Integrin alpha-V/beta-1 <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Integrin alpha-V/beta-1 (αVβ1) is a heterodimeric transmembrane receptor belonging to the integrin family, which mediates cell adhesion and signaling by interacting with extracellular matrix (ECM) proteins, such as fibronectin and vitronectin. This integrin plays a critical role in cellular processes such as migration, proliferation, and survival, particularly in development, wound healing, and angiogenesis. αVβ1 is involved in pathways like focal adhesion kinase (FAK) and PI3K/AKT signaling, linking ECM interactions to intracellular responses. Modulators of αVβ1, including inhibitors and allosteric regulators, can disrupt its binding to ECM ligands, thereby inhibiting angiogenesis and tumor progression. These modulators are under investigation for therapeutic applications in cancer, fibrosis, and other integrin-associated diseases.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Muscarinic acetylcholine receptor M1
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Muscarinic acetylcholine receptor M1 (M1 mAChR) is a G-protein-coupled receptor (GPCR) primarily expressed in the central nervous system, particularly in the brain's cortex and hippocampus. It is involved in the cholinergic signaling pathway, where its activation by acetylcholine regulates cognitive functions such as learning, memory, and attention, as well as neuronal excitability. M1 mAChR also mediates various physiological processes, including smooth muscle contraction and glandular secretion. Modulators of M1 mAChR, such as agonists and positive allosteric modulators, enhance acetylcholine signaling, showing potential for the treatment of cognitive disorders like Alzheimer's disease and schizophrenia. Antagonists, conversely, may be used to manage conditions like overactive bladder or motion sickness by inhibiting excessive cholinergic activity.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''

prompt_pk = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with providing a task description (a paragraph including definition, the factors that influence this property of a molecule (if any), what molecular properties are influenced by this property) for a given physico-chemical or ADMET property of small molecules. This paragraph should be concise and without nonsense (information not directly related to the three questions I asked); its content should be supported by established knowledge.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
logP
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
LogP, or the logarithmic partition coefficient, is a measure of a molecule's lipophilicity, calculated as the logarithm of the ratio of its concentrations in octanol and water at equilibrium. It is influenced by factors such as the presence of hydrophobic or hydrophilic functional groups, molecular size, and hydrogen bonding capacity. LogP directly impacts molecular properties including solubility, membrane permeability, and distribution in biological systems, making it critical in assessing a compound's pharmacokinetic and ADMET profiles.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
t1/2
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
t₁/₂, or half-life, is the time required for the concentration of a drug in the body to reduce by half. It is influenced by factors such as the rate of drug metabolism (determined by enzymatic activity), renal and hepatic clearance, and the drug's volume of distribution. Half-life affects molecular properties such as dosing frequency, duration of action, and steady-state concentration, making it a critical parameter in determining the pharmacokinetics and therapeutic regimen of a drug.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WodMol')
    parser.add_argument('--moldata', type=str, required=True, help='Dataset name (e.g., CHEMBL218)')
    parser.add_argument('--target', type=str, required=True, help='Target/property name (e.g., AKT)')
    parser.add_argument('--keyword', type=str, default='inhibitor', choices=['inhibitor', 'agonist', 'degrader', 'modulator', 'allosteric inhibitor', 'allosteric modulator', 'adme'], help='MOE of molecule (default: inhibitor)')
    parser.add_argument('--task_desc', type=str, default=None, help='User-specified task description file')
    parser.add_argument('--task_file', type=str, help="Name of the generated task embedding file (xx.npy)")
    parser.add_argument('--llm_path', type=str, help="Path of the llama file (/t9k/mnt/.cache/kagglehub/models/metaresearch/llama-3.1/transformers/8b-instruct/1)")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (default: cuda:0)')
    args = parser.parse_args()

    pro, func = args.target, args.keyword
    descmodel = joblib.load('dataset/raw/PCA1024model_task.pkl')
    if func == 'inhibitor':
        p = prompt_inh + pro + "<|eot_id|> " + "\n<|start_header_id|>assistant<|end_header_id|>"
    elif func == 'agonist':
        p = prompt_ago + pro + "<|eot_id|> " + "\n<|start_header_id|>assistant<|end_header_id|>" 
    elif func == 'modulator':
        p = prompt_mod + pro + "<|eot_id|> " + "\n<|start_header_id|>assistant<|end_header_id|>" 
    elif func == 'degrader':
        p = prompt_deg + pro + "<|eot_id|> " + "\n<|start_header_id|>assistant<|end_header_id|>" 
    elif func == 'allosteric inhibitor':
        p = prompt_ai + pro + "<|eot_id|> " + "\n<|start_header_id|>assistant<|end_header_id|>" 
    elif func == 'allosteric modulator':
        p = prompt_mi + pro + "<|eot_id|> " + "\n<|start_header_id|>assistant<|end_header_id|>" 
    elif func == 'adme':
        p = prompt_pk + pro + "<|eot_id|> " + "\n<|start_header_id|>assistant<|end_header_id|>" 
    else:
        if args.task_desc is None:
            raise ValueError('Give a correct keyword or task_desction file')

    if args.llm_path is None:
        raise ValueError("llm_path is required when task_desc is not provided")
    model_id = args.llm_path
    
    if args.task_desc is None:
        pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map=args.device)
        output = pipeline(p, max_new_tokens=500, return_full_text=False)[0]['generated_text']
        torch.cuda.empty_cache()
        output = output.strip()
        del pipeline
    else:
        with open(args.task_desc, 'r') as f:
            output = f.read()
            output = output.strip()
    print(output)
 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(args.device)
    inputs = tokenizer(output, return_tensors='pt').to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    sentence_embedding = last_hidden_state.mean(dim=1).cpu().numpy()
    
    embs = descmodel.transform(sentence_embedding)

    np.save(args.task_file, embs, allow_pickle=True)
