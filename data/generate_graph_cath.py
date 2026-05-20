import sys

sys.path.append("../")

import os
from dataloader.cath_dataset import Cath_imem, dataset_argument
from torch_geometric.data import Batch, Data
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import torch.nn.functional as F
import torch
from tqdm import tqdm
from utils import place_missing_cb, place_missing_o
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

amino_acids_name = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                    'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

amino_acids_multiple_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def processing_single(pdb_filename,save_dir,pdb_dir):
    try:
        graph = pdb2graph(pdb_dir + pdb_filename)
        if graph:
            torch.save(graph, save_dir + pdb_filename.replace('.pdb', '.pt'))
            return ("ok",pdb_filename,"")
        else:
            return ("error",pdb_filename,"Processing Error")
    except Exception as e:
        return ("error",pdb_filename,repr(e))


def get_struc2ndRes(pdb_filename):
    struc_2nds_res_alphabet = ['E', 'L', 'I', 'T', 'H', 'B', 'G', 'S']
    char_to_int = dict((c, i) for i, c in enumerate(struc_2nds_res_alphabet))

    p = PDBParser()
    structure = p.get_structure('random_id', pdb_filename)
    model = structure[0]
    dssp = DSSP(model, pdb_filename, dssp='mkdssp')

    # From model, extract the list of amino acids
    model_residues = [(chain.id, residue.id[1]) for chain in model for residue in chain if residue.id[0] == ' ']
    # From DSSP, extract the list of amino acids
    dssp_residues = [(k[0], k[1][1]) for k in dssp.keys()]

    # Determine the missing amino acids
    missing_residues = set(model_residues) - set(dssp_residues)

    # Initialize a list of integers for known secondary structures,
    # and another list of zeroes for one-hot encoding
    integer_encoded = []
    one_hot_list = torch.zeros(len(model_residues), len(struc_2nds_res_alphabet))

    current_position = 0
    for chain_id, residue_num in model_residues:
        dssp_key = (chain_id, (' ', residue_num, ' '))
        if (chain_id, residue_num) not in missing_residues and dssp_key in dssp:

            sec_structure_char = dssp[dssp_key][2]
            # L: coil
            sec_structure_char = sec_structure_char.replace('-', 'L')
            sec_structure_char = sec_structure_char.replace('P', 'L')  # for compatibility, new assigned structure 'P' in a new dssp version was mapped to 'L'
            integer_encoded.append(char_to_int[sec_structure_char])

            one_hot = F.one_hot(torch.tensor(integer_encoded[-1]), num_classes=8)
            one_hot_list[current_position] = one_hot
        else:
            print(pdb_filename, 'Missing residue: ', chain_id, residue_num, 'fill with 0')
        current_position += 1
    ss_encoding = one_hot_list[:current_position]
    return ss_encoding


def get_processed_graph(data):
    extra_x_feature = torch.cat([data.x[:, 20:], data.mu_r_norm], dim=1)
    graph = Data(
        x=data.x[:, :20],
        extra_x=extra_x_feature,
        pos=data.pos,
        atom_pos=data.atom_pos,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        ss=data.ss[:data.x.shape[0], :],
        sasa=data.x[:, 20]
    )
    return graph


def pdb2graph(filename):
    dataset_arg = dataset_argument()
    dataset = Cath_imem(dataset_arg['root'], dataset_arg['name'], split='test',
                        divide_num=dataset_arg['divide_num'], divide_idx=dataset_arg['divide_idx'],
                        c_alpha_max_neighbors=dataset_arg['c_alpha_max_neighbors'],
                        set_length=dataset_arg['set_length'],
                        struc_2nds_res_path=dataset_arg['struc_2nds_res_path'],
                        random_sampling=True, diffusion=True)
    rec, rec_coords, c_alpha_coords, n_coords, c_coords, o_coords, cb_coords, o_mask, cb_mask = dataset.get_receptor_inference(
        filename)
    struc_2nd_res = get_struc2ndRes(filename)
    rec_graph = dataset.get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, o_coords, cb_coords, rec_coords,
                                         struc_2nd_res)
    # atom_pos = [n_coords, c_alpha_coords, c_coords, cb_coords, o_coords]

    rec_graph.atom_pos = place_missing_cb(rec_graph.atom_pos)
    rec_graph.atom_pos = place_missing_o(rec_graph.atom_pos, o_mask)

    if rec_graph:
        return rec_graph
    else:
        return None

# improved for multiprocessing
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--num-workers', type=int, default=4)
    args=parser.parse_args()
    # generate a batch of protein graph
    error_pdb = []
    place_virtual_cb = True
    for key in ['test', 'validation', 'train']:
        pdb_dir = f'cath_download/{key}/'
        save_dir = f'cath/cath_process/{key}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        task_list = [i for i in os.listdir(pdb_dir) if i.endswith('.pdb') and not os.path.exists(save_dir + i.replace('.pdb', '.pt'))]

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures=[executor.submit(processing_single, pdb_filename,save_dir,pdb_dir) for pdb_filename in task_list]

            for future in tqdm(as_completed(futures),total=len(futures)):
                if future.result()[0] == "error":
                    error_pdb.append([future.result()[1],future.result()[2]])

                print(f"{future.result()[1]} status: {future.result()[0]} {future.result()[2]}")

    print(f"Num of Failed: {len(error_pdb)}")
    print(f"Failed pdbs: {error_pdb}")
