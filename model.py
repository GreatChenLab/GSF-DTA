import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SequenceEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class GSF_DTA(torch.nn.Module):
    def __init__(self, protein_in_channels, drug_in_channels, protein_seq_dim, drug_seq_dim, hidden_channels, out_channels):
        super(GSF_DTA, self).__init__()
        self.protein_conv1 = GCNConv(protein_in_channels, hidden_channels)
        self.protein_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.drug_conv1 = GCNConv(drug_in_channels, hidden_channels)
        self.drug_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.protein_seq_encoder = SequenceEncoder(protein_seq_dim, hidden_channels)
        self.drug_seq_encoder = SequenceEncoder(drug_seq_dim, hidden_channels)

        self.fc1 = torch.nn.Linear(4 * hidden_channels, hidden_channels)
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
