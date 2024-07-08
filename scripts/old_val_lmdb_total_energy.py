from ocpmodels.preprocessing import AtomsToGraphs
from multiprocessing import Pool

import lmdb
import ase
import pickle
import os
import numpy as np
import torch
from tqdm import tqdm

## load the mappings stored from previous experiments
with open('../adslab_tags_full.pkl', 'rb') as read_file:
    tags_mappings = pickle.load(read_file)

a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=False,    # False for test data
    r_forces=False,    # False for test data
    r_distances=False,
    r_fixed=True,
)




def write_lmdbs(mp_args):
    lmdb_idx, train_trajs_split = mp_args
    idx = 0
    val_db = lmdb.open(
    f"/home/jovyan/shared-scratch/kabdelma/oc20_data_quality_project/s2ef_200k_experiment/lmdbs/total_energy/val_30k/old_dft_data_all/{lmdb_idx}.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
    )
    for traj_file in tqdm(train_trajs_split):
        # get the atoms obj
        sid = traj_file.split("_")[0]
        fid = traj_file.split("_")[1].split(".")[0]
        old_atoms_object = ase.io.read(f"/home/jovyan/shared-datasets/OC20/trajs/val_02_01/{sid}.traj", fid)
        image = a2g.convert(old_atoms_object)
        image.y = old_atoms_object.get_potential_energy()
        image.force  = torch.tensor(old_atoms_object.get_forces())
        image.sid = torch.LongTensor([int(sid[6:])]) # get the integer in the sid
        image.fid = torch.LongTensor([int(fid)])
        image.tags = torch.LongTensor(tags_mappings[sid])
        # Write to LMDB
        txn = val_db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(image, protocol=-1))
        txn.commit()
        val_db.sync()
        idx += 1


    txn = val_db.begin(write=True)
    txn.put(f"length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    val_db.sync()
    val_db.close()


train_trajs = os.listdir("/home/jovyan/shared-scratch/kabdelma/oc20_data_quality_project/s2ef_200k_experiment/oc20_theory_expts/val-30k/clean_outputs")
# Create a multiprocessing Pool
NUM_WORKERS = 16
# split up the dictionary that has all the atoms objects
train_trajs_splits = np.array_split(train_trajs, NUM_WORKERS)
# pool over these splits 
pool = Pool(NUM_WORKERS)     
mp_args = [(lmdb_idx, subsplit) for lmdb_idx, subsplit in enumerate(train_trajs_splits)]
list(pool.imap(write_lmdbs, mp_args))
