import pandas as pd
from torch.utils.data import Dataset
from protein_graph import protein_to_graph
from drug_graph import smiles_to_graph
import torch
import esm

def pad_protein_sequence(seq, max_length=1000):
    if len(seq) < max_length:
        seq = seq.ljust(max_length, '<pad>')
    elif len(seq) > max_length:
        seq = seq[:max_length]
    return seq

def pad_drug_sequence(seq, max_length=128):
    if len(seq) < max_length:
        seq = seq.ljust(max_length, '<pad>')
    elif len(seq) > max_length:
        seq = seq[:max_length]
    return seq

def build_drug_vocab(data):
    vocab = set()
    for smiles in data:
        for char in smiles:
            vocab.add(char)
    vocab = sorted(vocab)
    vocab_dict = {char: idx for idx, char in enumerate(vocab)}
    return vocab_dict
    
def encode_drug_sequence(seq, vocab):
    encoded = [vocab[char] if char in vocab else vocab['<unk>'] for char in seq]
    return torch.tensor(encoded, dtype=torch.long)

def encode_protein_sequence(seq):
    alphabet = esm.data.Alphabet.from_architecture('ESM-2')
    batch_converter = alphabet.get_batch_converter()
    _, _, tensor = batch_converter([('protein', seq)])
    return tensor

class DavisDataset(Dataset):
    def __init__(self, csv_file, alpha_fold_dir, has_labels=True):
        self.data = pd.read_csv(csv_file)
        self.has_labels = has_labels
        self.alpha_fold_dir = alpha_fold_dir
        self.drug_vocab = build_drug_vocab(self.data['Drug SMILES'])
        self.drug_vocab['<pad>'] = len(self.drug_vocab)
        self.drug_vocab['<unk>'] = len(self.drug_vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uniprot_id = self.data.iloc[idx, 0]
        drug_smiles = self.data.iloc[idx, 1]
        protein_seq = self.data.iloc[idx, 2]
        drug_seq = self.data.iloc[idx, 1] 

        protein_seq = pad_protein_sequence(protein_seq)
        drug_seq = pad_drug_sequence(drug_seq)

        protein_graph = protein_to_graph(uniprot_id, self.alpha_fold_dir)

        drug_graph = smiles_to_graph(drug_smiles)

        protein_seq_encoded = encode_protein_sequence(protein_seq)
        drug_seq_encoded = encode_drug_sequence(drug_seq, self.drug_vocab)

        if self.has_labels:
            label = torch.tensor(self.data.iloc[idx, 4], dtype=torch.float32)  # 假设第五列是亲和力标签
            return protein_graph, drug_graph, protein_seq_encoded, drug_seq_encoded, label
        return protein_graph, drug_graph, protein_seq_encoded, drug_seq_encoded
