import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import GSF_DTA
from metrics import rmse, pearson
from tqdm import tqdm


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=1000, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_val_rmse = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}') as pbar:
            for prot_g, drug_g, prot_seq, drug_seq, labels in pbar:
                prot_g = prot_g.to(device)
                drug_g = drug_g.to(device)
                prot_seq = prot_seq.to(device)
                drug_seq = drug_seq.to(device)
                labels = labels.to(device).view(-1, 1)

                optimizer.zero_grad()
                outputs = model(prot_g.x, prot_g.edge_index, drug_g.x, drug_g.edge_index, prot_seq, drug_seq)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

        model.eval()
        val_rmse, val_pearson = evaluate_model(model, val_loader, device, criterion)

        print(f'\nEpoch {epoch}, Train Loss: {total_loss / len(train_loader):.4f}, '
              f'Val RMSE: {val_rmse:.4f}, Pearson: {val_pearson:.4f}')

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), 'best_davis_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    model.load_state_dict(torch.load('best_davis_model.pth'))
    return model


def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    preds = []
    labels = []
    total_loss = 0.0
    with torch.no_grad():
        for prot_g, drug_g, prot_seq, drug_seq, label in data_loader:
            prot_g = prot_g.to(device)
            drug_g = drug_g.to(device)
            prot_seq = prot_seq.to(device)
            drug_seq = drug_seq.to(device)
            label = label.to(device).view(-1, 1)

            outputs = model(prot_g.x, prot_g.edge_index, drug_g.x, drug_g.edge_index, prot_seq, drug_seq)
            total_loss += criterion(outputs, label).item()

            preds.extend(outputs.cpu().numpy())
            labels.extend(label.cpu().numpy())

    val_rmse = rmse(np.array(labels), np.array(preds))
    val_pearson = pearson(np.array(labels), np.array(preds))
    return val_rmse, val_pearson
