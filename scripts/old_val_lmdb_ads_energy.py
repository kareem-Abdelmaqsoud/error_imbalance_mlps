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
LMDB_PATH = "/home/jovyan/shared-scratch/kabdelma/OC20_dataset_errors/train_single_pts/lmdbs/ads_energy/val_30k/old_dft_data_all"
NEW_ADSLABS_PATH = "/home/jovyan/shared-scratch/kabdelma/OC20_dataset_errors/train_single_pts/oc20_theory_expts/val-30k/clean_outputs/"
NEW_SLAB_PATH = "/home/jovyan/shared-scratch/kabdelma/oc20_surface_calcs/"
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
    r_edges=False,
    r_forces=False,    # False for test data
    r_distances=False,
    r_fixed=True,
)

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
        # get the sid and the fid from the traj name. Ex: random999878_100.traj
        sid = traj_file.split("_")[0]
        fid = traj_file.split("_")[1].split(".")[0]
#         if sid + ".traj" in slab_trajs:
#             if mappings[sid]["class"]==2:
        old_atoms_object = ase.io.read(f"{OLD_ADSLABS_PATH}/{sid}.traj", fid)
        image = a2g.convert(old_atoms_object)
        # get the adsorption energy
        image.y = old_atoms_object.get_potential_energy() - ref_energy_mappings[f"{sid}"]
        # get the total energy
#             image.y_total = old_atoms_object.get_potential_energy()
        image.force  = torch.tensor(old_atoms_object.get_forces())
        # get the integer in the sid
        image.sid = torch.LongTensor([int(sid[6:])]) 
        image.fid = torch.LongTensor([int(fid)])
        # get the atoms tags
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
    
adslab_trajs = os.listdir(NEW_ADSLABS_PATH)
slab_trajs = os.listdir(NEW_SLAB_PATH)
# split up the dictionary that has all the atoms objects
train_trajs_splits = np.array_split(adslab_trajs, NUM_WORKERS)
# pool over these splits 
pool = Pool(NUM_WORKERS)     
mp_args = [(lmdb_idx, subsplit) for lmdb_idx, subsplit in enumerate(train_trajs_splits)]
list(pool.imap(write_lmdbs, mp_args))