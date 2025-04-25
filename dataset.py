import pandas as pd
from torch.utils.data import Dataset
from protein_graph import protein_to_graph
from drug_graph import smiles_to_graph


class DavisDataset(Dataset):
    def __init__(self, csv_file, alpha_fold_dir, has_labels=True):
        self.data = pd.read_csv(csv_file)
        self.has_labels = has_labels
        self.alpha_fold_dir = alpha_fold_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uniprot_id = self.data.iloc[idx, 0]
        drug_smiles = self.data.iloc[idx, 1]

        protein_graph = protein_to_graph(uniprot_id, self.alpha_fold_dir)

        drug_graph = smiles_to_graph(drug_smiles)

        if self.has_labels:
            label = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)
            return protein_graph, drug_graph, label
        return protein_graph, drug_graph