# CondMol

![](img/Framework.jpg)

This is a Pytorch implementation of CondMol
Information regarding the activity pre-training dataset can be found in the `dataset/` folder.
Pre-trained model parameters can be downloaded from [Google Drive](https://drive.google.com/file/d/1BjL2PjrnE0UVObQKFWRz8Oioe-rbkZ21/view?usp=sharing), then extracted to the `checkpoints/` folder:
* **`model.pkl`**: Trained without considering censored data information.
* **`model_CSLoss.pkl`**: Trained incorporating censored data information.
* **`model_noleakage.pkl`**: The pre-training dataset excludes assay data related to the 29 targets in CondACT (to avoid data leakage).

## Installation

### Prerequisites
Please ensure your hardware (specifically GPU VRAM) is sufficient to support running **Llama-3.1-8B-Instruct** at a minimum. 
* **Minimum:** Support for the 8B model.
* **Recommended:** We strongly encourage using larger versions of the model for superior performance.
* **Compatibility:** While optimized for Llama-3.1, you are welcome to attempt using other LLMs.

### Setup
You can execute the following command to create the conda environment:
```
conda create --name ConMol --file requirements.yml
conda activate CondMol
```
You Need to Get Access to Llama-3.1 from HuggingFace


## Usage

### 1. Dataset Preparation
#### a. Generate Task Embeddings
Before training, you need to generate a task embedding file (`.npy`). You have two options to generate the description:
1.  **LLM Generation (Default)**: Provide the target name and a keyword (MOA or 'adme'). The script will use Llama-3.1 to generate a scientific description.
2.  **Custom Description**: Provide a `.txt` file containing your own description using `--task_desc`.

Use this for standard protein targets (targetname + MOA) or ADME properties (propertyname + adme).
```
python run.py --mode taskemb \
    --moldata CHEMBL218 \               # Dataset name
    --target "Cannabinoid receptor 1" \ # Target name (e.g., "AKT1") or Property name (e.g., "Solubility")
    --keyword inhibitor \               # MOA: inhibitor, agonist, degrader, modulator, allosteric inhibitor/modulator, or "adme"
    --task_file tasks_AKT.npy \         # Output path for the generated embedding (.npy)
    --llm_path /t9k/mnt/.cache/kagglehub/models/metaresearch/llama-3.1/transformers/8b-instruct/1 \        # Path of the llama file
    --device cuda:0                     # Device to run the LLM
```

Use this if you have a specific description text file (e.g., desc.txt).
```
python run.py --mode taskemb \
    --moldata MyData \                 # Dataset name
    --target UnknownTarget \           # Required argument (placeholder name is fine here)
    --task_desc desc.txt \             # Path to your custom description text file
    --task_file tasks_custom.npy \     # Output path for the generated embedding (.npy)
    --llm_path /t9k/mnt/.cache/kagglehub/models/metaresearch/llama-3.1/transformers/8b-instruct/1 \        # Path of the llama file
    --device cuda:0                    # Device to run the embedding model
```

#### b. Generate Condition Embeddings & Preprocessing

**Step 1: Prepare Raw Data**
Place your files (`{name}_train.csv`, `{name}_test.csv`) in `dataset/raw/`.
* **Required Columns:** `canonical_smiles`, `bao_label`, `standard_type`, `standard_units`, `assay_organism`, `assay_tissue`, `standard_relation`, `assay_chembl_id`, `assay_description`, `assay_cell_type`. Note: Unknown fields can be left as empty values.
* **Label Processing:** Convert concentration units (IC50/Ki) to **Molar** and take **negative log** (e.g., pIC50); convert percentages to decimals.
* **Censored Data:** Ensure `standard_relation` contains (`>`, `<`, `=`). **Note:** If you applied negative log, invert the signs (e.g., `>` becomes `<`).

**Step 2: Generate Conditions**
Run `run.py --mode condemb` to extract experimental conditions using the LLM (generates `{name}_train_llamacond.csv` `{name}_test_llamacond.csv`).

```
python run.py --mode condemb \
    --moldata CHEMBL218 \              # Dataset name (Processing {name}_train.csv and {name}_test.csv)
    --device cuda:0 \                  # Device to run the extraction model
    --testing False                    # Set True if you only want to process the test/zeroshot dataset
    --llm_path /t9k/mnt/.cache/kagglehub/models/metaresearch/llama-3.1/transformers/8b-instruct/1 \        # Path of the llama file
```
> [!IMPORTANT]
> **🔥 Advanced: Manual Mode (Skip Step 2)**
> You can skip the LLM extraction by directly providing prepared `{name}_train_llamacond.csv` and `{name}_test_llamacond.csv` files.
>
> **Required Columns:**
> `Smiles`, `bao_label`, `standard_type`, `standard_units`, `assay_organism`, `assay_tissue`, `Solvents`, `Incubation Time`, `Temperature`, `pH`, `Compound Concentration`, `Assay Method`, `Compound Administration Method`, `Cell Line`, `standard_relation`, `assay_chembl_id`, `assay_description`.

> [!TIP]
> **⚡ Optimization Tip**
> The LLM extraction process can be slow. To improve efficiency, for **data batches sharing identical experimental conditions**, you only need to extract conditions for **one representative sample** and copy them to all other samples in the same batch.

#### c. Preprocess
Preprocess Graph Data Convert the condition-annotated CSV files into PyTorch Geometric (.pt) files in `dataset/processed/`.
```
python run.py --mode preprocess \
    --moldata CHEMBL218 \              # Dataset name
    --numtasks 1 \                     # Number of tasks/labels
    --label_col standard_value \       # Name of label column(s). For multi-task, use commas to split: IC50,EC50,Ki
    --device cuda:0 \                  # Device
    --testing False                    # Set True to process only the test dataset
    --llm_path /t9k/mnt/.cache/kagglehub/models/metaresearch/llama-3.1/transformers/8b-instruct/1 \        # Path of the llama file
```


### 2. Fine-tuning on Downstream Tasks

To fine-tune the model on specific tasks, **please first ensure your training&testing datasets and taskembeding file are prepared following the instructions in Section 1**. Then, configure the `pi` (task ID) and `rela` (relation) arguments based on your task type:

* **For Known Tasks (Present in Pre-training Dataset):**
    * Refer to `dataset/task_stats_report.yaml` to identify your target task.
    * Locate the corresponding `task_name` and `task_type` (Note: protein activity tasks are listed first, followed by representative assay descriptions for non-protein tasks).
    * Set the argument `pi` to the corresponding **task_id**.

* **For Unknown Tasks (New/Unseen Datasets):**
    * Set the argument `pi` to `None`.
    * **Recommendation:** We suggest treating all **non-protein tasks** as unknown tasks (`pi=None`).

* **Handling Censored Data:**
    * If your dataset contains censored information (e.g., values with `>` or `<`), set the argument `rela` to `True`.
    * Ensure that the `standard_relation` column is preserved in your raw input CSV file to utilize this feature.


Once you have determined your task settings, you can execute the fine-tuning script.
The following is an example command for fine-tuning on the C218 dataset. **Note that you can adjust the fine-tuning hyperparameters according to your specific task requirements:

```
python run.py --mode finetune \
    --moldata CHEMBL218 \                     # Dataset name (Look for {name}_train.csv in dataset/)
    --pretrain checkpoints/model_CSLoss.pkl \ # Path to pre-trained weights
    --pi None \                               # Task ID: Integer for known tasks, None for unknown/non-protein
    --rela False \                            # Relation: Set True if dataset has censored data (>, <)
    --task_file CHEMBL218.npy \               # Path to task embedding file
    --train_epoch 30 \                        # Number of training epochs
    --batch_size 32 \                         # Input batch size
    --lr 1e-3 \                               # Learning rate
    --dropout 0.1 \                           # Dropout ratio
    --seed 426 \                              # Random seed for reproducibility
    --fold 5 \                                # Number of folds
    --device cuda:0 \                         # Specify which GPU to use
    --savem False                             # Set True to save the best model checkpoints
```
* --savem: Set to `True` to save model checkpoints during training. **The model parameters will be saved to the `log/checkpoint/` directory.**
You are encouraged to adjust these parameters to optimize performance for your specific dataset.


To reproduce the experimental results reported in the paper, you can execute the following scripts. 
The datasets are provided in the `CondACT`, `CondACT_few`, and `CondADME` folders [Google Drive](https://drive.google.com/drive/folders/1Zm6LMLDmZv7K9WyxgJR78nC1DiizkHRX?usp=sharing). Run the following commands to reproduce the experiments:

```
python run.py --mode condact --moldata <DATASET_NAME>
python run.py --mode condactfew --moldata <DATASET_NAME>
python run.py --mode condadme --moldata <DATASET_NAME>
```

### 3. Testing / Zero-shot Prediction

**Prerequisites:**
Before running prediction, please ensure that your testing datasets and taskembeding file are prepared following the instructions in Section 1

**Task Configuration (`pi`):**
* **Zero-shot / Unknown Task:** Set `--pi None`.
* **Known Task:** Set `--pi` to the corresponding Task ID.

**Execution Command:**

```
python run.py --mode zeroshot \
    --moldata CHEMBL218 \                     # Dataset name (Look for {name}_test.csv in dataset/)
    --pretrain checkpoints/model_CSLoss.pkl \ # Path to pre-trained weights (model_CSLoss.pkl is recommended) or fine-tuned checkpoint file saved in Section 2
    --pi None \                               # Task ID: Integer for known tasks, None for unknown/Zero-shot
    --ft False \                              # Set True if you are using a fine-tuned checkpoint
    --task_file tasks.npy \                   # Path to the task embedding file (.npy)
    --batch_size 32 \                         # Input batch size
    --seed 426 \                              # Random seed for reproducibility
    --device cuda:0                           # Specify which GPU to use
```


## Citation




