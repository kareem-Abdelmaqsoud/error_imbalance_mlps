from ocpmodels.preprocessing import AtomsToGraphs
from multiprocessing import Pool

import lmdb
import ase
import pickle
import os
import numpy as np
import torch
from tqdm import tqdm

## PATHS
NEW_ADSLABS_PATH = "/home/jovyan/shared-scratch/kabdelma/oc20_data_quality_project/s2ef_200k_experiment/oc20_theory_expts/val-30k/clean_outputs"
NEW_SLAB_PATH = "/home/jovyan/shared-scratch/kabdelma/oc20_data_quality_project/oc20_surface_calcs"
OLD_ADSLABS_PATH = "/home/jovyan/shared-datasets/OC20/trajs/val_02_01"
# Create a multiprocessing Pool
NUM_WORKERS = 16

## load the mappings stored from previous experiments
with open('../adslab_tags_full.pkl', 'rb') as read_file:
    tags_mappings = pickle.load(read_file)
    
# mapping to convert adsorption energy to total energies
with open("../oc20_ref.pkl", "rb") as input_file:
    ref_energy_mappings = pickle.load(input_file)

# get the system information mappings to select the nonmetallic materials
with open("../../oc20_data_mapping.pkl", "rb") as input_file:
    mappings = pickle.load(input_file)

a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=False,    # False for test data
    r_forces=False,    # False for test data
    r_distances=False,
    r_edges=False,
    r_fixed=True,
)


def compute_ref_energy(sid, adslab_atoms_object):
    ## compute the energy of the adsorbate at the new DFT settings
    # identify the adsorbate atoms 
    adsorbate_index = np.where(tags_mappings[sid]==2)[0]
    # mapping of the atomic numbers to the energy of each atom at the new settings
    ads_ref_energies = { 7: -8.133240085, 1: -3.492536275, 8: -7.15771642, 6: -7.26043086}
    adsorbate_energy = np.sum([ads_ref_energies[atom_num] for atom_num in adslab_atoms_object.numbers[adsorbate_index]])
    ## get the slab energy at the new settings
    slab_atoms_obj = ase.io.read(f"{NEW_SLAB_PATH}/{sid}.traj")
    slab_energy = slab_atoms_obj.get_potential_energy()
    # reference energy = slab_energy + adsorbate energy
    ref_energy = slab_energy + adsorbate_energy
    return ref_energy


def write_lmdbs(mp_args):
    lmdb_idx, train_trajs_split = mp_args
    idx = 0
    train_db = lmdb.open(
    f"{LMDB_PATH}/s2ef_{lmdb_idx}.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
    )
    for traj_file in tqdm(train_trajs_split):
        sid = traj_file.split("_")[0]
        fid = traj_file.split("_")[1].split(".")[0]
        if sid + ".traj" in slab_trajs:
            # get the atoms obj
            atoms = ase.io.read(f"{NEW_ADSLABS_PATH}/{traj_file}")
            image = a2g.convert(atoms)
             # get the adsorption energy
            ref_energy = atoms.get_potential_energy() - compute_ref_energy(sid,atoms)
            # if mappings[sid]["class"]==2:
            #     np.random.seed(0)
            #     ref_energy = ref_energy + np.random.normal(mu, sigma)
            image.y = ref_energy
            image.y_total = atoms.get_potential_energy()
            image.force  = torch.tensor(atoms.get_forces())
            # get the integer in the sid
            image.sid = torch.LongTensor([int(sid[6:])]) 
            image.fid = torch.LongTensor([int(fid)])
            image.tags = torch.LongTensor(tags_mappings[sid])
            # Write to LMDB
            txn = train_db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(image, protocol=-1))
            txn.commit()
            train_db.sync()
            idx += 1

    txn = train_db.begin(write=True)
    txn.put(f"length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    train_db.sync()
    train_db.close()
    
mu, sigma = 0, 0.1
LMDB_PATH = "/home/jovyan/shared-scratch/kabdelma/oc20_data_quality_project/s2ef_200k_experiment/lmdbs/val_30k_noise/sigma_0.1"

adslab_trajs = os.listdir(NEW_ADSLABS_PATH)
slab_trajs = os.listdir(NEW_SLAB_PATH)
# split up the dictionary that has all the atoms objects
train_trajs_splits = np.array_split(adslab_trajs, NUM_WORKERS)
# pool over these splits 
pool = Pool(NUM_WORKERS)     
mp_args = [(lmdb_idx, subsplit) for lmdb_idx, subsplit in enumerate(train_trajs_splits)]
list(pool.imap(write_lmdbs, mp_args))