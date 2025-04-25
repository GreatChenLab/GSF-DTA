import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GSF_DTA(torch.nn.Module):
    def __init__(self, protein_in_channels, drug_in_channels, hidden_channels, out_channels):
        super(GSF_DTA, self).__init__()
        self.protein_conv1 = GCNConv(protein_in_channels, hidden_channels)
        self.protein_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.drug_conv1 = GCNConv(drug_in_channels, hidden_channels)
        self.drug_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, protein_x, protein_edge_index, drug_x, drug_edge_index):
        protein_x = self.protein_conv1(protein_x, protein_edge_index)
        protein_x = F.relu(protein_x)
        protein_x = self.protein_conv2(protein_x, protein_edge_index)
        protein_x = F.relu(protein_x)
        protein_x = protein_x.mean(dim=0)

        drug_x = self.drug_conv1(drug_x, drug_edge_index)
        drug_x = F.relu(drug_x)
        drug_x = self.drug_conv2(drug_x, drug_edge_index)
        drug_x = F.relu(drug_x)
        drug_x = drug_x.mean(dim=0)

        combined_x = torch.cat([protein_x, drug_x], dim=0)
        combined_x = self.fc1(combined_x)
        combined_x = F.relu(combined_x)
        output = self.fc2(combined_x)
        return output