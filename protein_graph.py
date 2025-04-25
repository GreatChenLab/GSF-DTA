import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial.distance import cdist
from alphafold_utils import download_alphafold_structure


def get_atom_coordinates(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    atom_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coords.append(atom.get_coord())
    return np.array(atom_coords)


def protein_to_graph(uniprot_id, alpha_fold_dir, cutoff=10.0):
    pdb_file = download_alphafold_structure(uniprot_id, alpha_fold_dir)
    if not pdb_file:
        raise ValueError(f"Failed to get PDB file for {uniprot_id}")

    atom_coords = get_atom_coordinates(pdb_file)
    num_atoms = len(atom_coords)

    node_features = torch.ones((num_atoms, 1), dtype=torch.float32)

    dist_matrix = cdist(atom_coords, atom_coords)
    edge_mask = (dist_matrix <= cutoff) & (dist_matrix > 1e-6)
    edge_index = np.argwhere(edge_mask).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index)