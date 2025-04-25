# GSF-DTA
![Figure 1](https://github.com/user-attachments/assets/2dcb2ecb-0e7d-4124-8b5e-94150f1cb5f7)


## Directory Structure

```python
dta_project/
├── alphafold_utils.py  # AlphaFold structure download utility
├── drug_graph.py       # Drug graph construction module
├── metrics.py          # Evaluation metrics
├── model.py            # Model definition
├── protein_graph.py    # Protein graph construction (with AlphaFold support)
├── dataset.py          # Dataset loading module
├── train.py            # Training logic
├── test.py             # Testing logic
├── main.py             # Main program entry
└── requirements.txt    # Dependency list
```
## Must installed packages or softwares

```python
torch==1.10.1
torch_geometric==2.0.4
rdkit-pypi==2023.3.3
biopython==1.79
pandas==1.3.3
scipy==1.7.1
requests==2.26.0
tqdm==4.64.0
numpy==1.21.2
```

## Data Preparation

```python
data/Davis/DTA/
├── train.csv
└── test.csv

data/KIBA/DTA/
├── train.csv
└── test.csv

data/BindingDB/DTA/
├── train.csv
└── test.csv
```
## Usage Steps
### 1. Environment Setup
```python
# Create and activate a conda environment
conda create -n dta_env python=3.7
conda activate dta_env

# Install dependencies
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
pip install torch_geometric==2.0.4 torch-scatter==2.0.9 torch-sparse==0.6.17 torch-spline-conv==1.2.10 -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install rdkit-pypi==2023.3.3 biopython==1.79 pandas==1.3.3 scipy==1.7.1 requests==2.26.0 tqdm==4.64.0 numpy==1.21.2
```
### 2. Download AlphaFold Structures
The model automatically downloads PDB files from the AlphaFold database. Ensure network connectivity.

 ### 3. Train the Model

```python
# Run after training
python test.py
```
The training process includes early stopping (halts if validation loss does not decrease for 10 consecutive epochs) and saves the best model.
### 4. Test the Model

```python
# Run after training
python test.py
```

