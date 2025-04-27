import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import esm

class ProteinSequenceEncoder(nn.Module):
    def __init__(self):
        super(ProteinSequenceEncoder, self).__init__()
        self.model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, protein_seq):
        results = self.model(protein_seq, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        return token_representations[:, 0, :]

class DrugSequenceEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, max_length=128):
        super(DrugSequenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.max_length = max_length

    def forward(self, drug_seq):
        embedded = self.embedding(drug_seq)
        return embedded.mean(dim=1)

class GSF_DTA(torch.nn.Module):
    def __init__(self, protein_in_channels, drug_in_channels, hidden_channels, out_channels):
        super(GSF_DTA, self).__init__()
        self.protein_conv1 = GCNConv(protein_in_channels, hidden_channels)
        self.protein_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.drug_conv1 = GCNConv(drug_in_channels, hidden_channels)
        self.drug_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.protein_seq_encoder = ProteinSequenceEncoder()
        self.drug_seq_encoder = DrugSequenceEncoder(vocab_size=100)  # 这里假设词汇表大小为 100，可根据实际修改

        self.fc1 = torch.nn.Linear(2 * hidden_channels + 1280 + 128, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, protein_x, protein_edge_index, drug_x, drug_edge_index, protein_seq, drug_seq):
        protein_graph_x = self.protein_conv1(protein_x, protein_edge_index)
        protein_graph_x = F.relu(protein_graph_x)
        protein_graph_x = self.protein_conv2(protein_graph_x, protein_edge_index)
        protein_graph_x = F.relu(protein_graph_x)
        protein_graph_x = protein_graph_x.mean(dim=0)

        drug_graph_x = self.drug_conv1(drug_x, drug_edge_index)
        drug_graph_x = F.relu(drug_graph_x)
        drug_graph_x = self.drug_conv2(drug_graph_x, drug_edge_index)
        drug_graph_x = F.relu(drug_graph_x)
        drug_graph_x = drug_graph_x.mean(dim=0)

        protein_seq_x = self.protein_seq_encoder(protein_seq)
        drug_seq_x = self.drug_seq_encoder(drug_seq)

        combined_x = torch.cat([protein_graph_x, drug_graph_x, protein_seq_x, drug_seq_x], dim=0)
        combined_x = self.fc1(combined_x)
        combined_x = F.relu(combined_x)
        output = self.fc2(combined_x)
        return output
