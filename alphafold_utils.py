import os
import requests
import time

ALPHAFOLD_DB_URL = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"


def download_alphafold_structure(uniprot_id, save_dir):
    """
    从AlphaFold数据库下载蛋白质结构的PDB文件
    :param uniprot_id: UniProt ID
    :param save_dir: 保存PDB文件的目录
    :return: 下载成功返回PDB文件路径，失败返回None
    """
    os.makedirs(save_dir, exist_ok=True)
    pdb_url = ALPHAFOLD_DB_URL.format(uniprot_id)
    pdb_file_path = os.path.join(save_dir, f"{uniprot_id}.pdb")

    if os.path.exists(pdb_file_path):
        print(f"PDB file for {uniprot_id} already exists. Skipping download.")
        return pdb_file_path

    try:
        response = requests.get(pdb_url)
        if response.status_code == 200:
            with open(pdb_file_path, 'w') as f:
                f.write(response.text)
            print(f"Successfully downloaded PDB file for {uniprot_id}.")
            return pdb_file_path
        else:
            print(f"Failed to download PDB file for {uniprot_id}. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Request error occurred while downloading PDB file for {uniprot_id}: {e}")
        return None
