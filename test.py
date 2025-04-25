import torch
import pandas as pd
from model import GSF_DTA
from metrics import *


def test_model(model, test_loader, device, output_file='davis_test_predictions.csv'):
    model.eval()
    model.to(device)
    preds = []
    labels = []

    with torch.no_grad():
        for prot_g, drug_g, *rest in test_loader:
            prot_g = prot_g.to(device)
            drug_g = drug_g.to(device)
            outputs = model(prot_g.x, prot_g.edge_index, drug_g.x, drug_g.edge_index)
            preds.extend(outputs.cpu().numpy())
            if rest:  
                labels.extend(rest[0].cpu().numpy())

    test_data = pd.read_csv('data/Davis/DTA/test.csv')
    test_data['predicted_affinity'] = preds
    test_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    if labels:
        metrics = {
            'RMSE': rmse(np.array(labels), np.array(preds)),
            'MSE': mse(np.array(labels), np.array(preds)),
            'Pearson': pearson(np.array(labels), np.array(preds)),
            'Spearman': spearman(np.array(labels), np.array(preds)),
            'CI': ci(np.array(labels), np.array(preds))
        }
        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    return test_data