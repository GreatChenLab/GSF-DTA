import torch
from dataset import DavisDataset
from torch.utils.data import DataLoader
from model import GSF_DTA
from train import train_model, evaluate_model
from test import test_model

DATA_PATH = 'data/Davis'
TRAIN_CSV = DATA_PATH + '_train.csv'
TEST_CSV = DATA_PATH + '_test.csv'
ALPHAFOLD_DIR = 'alphafold_pdb_files'


def main():
    train_dataset = DavisDataset(TRAIN_CSV, ALPHAFOLD_DIR)
    test_dataset = DavisDataset(TEST_CSV, ALPHAFOLD_DIR)

    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = GSF_DTA(protein_in_channels=1, drug_in_channels=41, hidden_channels=64, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    trained_model = train_model(model, train_loader, val_loader, optimizer, criterion)

    test_model(trained_model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == __main__
    main()