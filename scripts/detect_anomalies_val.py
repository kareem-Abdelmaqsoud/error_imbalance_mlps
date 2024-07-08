## import modules 
from ocdata.utils.vasp import write_vasp_input_files
from ocpmodels.datasets import LmdbDataset as LD
from ocpmodels.common.relaxation.ase_utils import batch_to_atoms
from ocpmodels.common.data_parallel import ParallelCollater
from torch.utils.data import DataLoader
from ocdata.utils import DetectTrajAnomaly
# import libraries
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import ase
import ase.io
import os

LMDB_PATH =  "/home/jovyan/shared-datasets/OC20/s2ef/all/val_id_30k/"
BATCH_SIZE = 16
NUM_WORKERS = 8


# here we loop over them and create a dictionary of this strcuture {sid_fid:{"atoms":atoms_obj}} 
dataset = LD({"src": LMDB_PATH})
collate_fn = ParallelCollater(0, True)
# load systems in batches 
loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            batch_size=BATCH_SIZE,
)

# parse the lmdbs to get the atoms objects, sids and fids
ref_energies = []
sids_numbers = []
fids_numbers = []
for batch in tqdm(loader):
    # convert a batch of graphs to atoms objects
    ref_energies += batch[0].y
    sids_numbers += batch[0].sid
    fids_numbers += batch[0].fid

## get the fids and sids in the correct format and outside the tensors
sids_fids = []
sids = []
fids = []
ref_energies_items = []
for (sid_num, fid, energy) in zip(sids_numbers, fids_numbers, ref_energies):
    # becuase one system can have multiple frames
    # combine system id + frame id "sid#_fid#" to define unique frames
    ref_energies_items.append(energy.item())
    sids.append("random" + str(sid_num.item()))
    fids.append(fid.item())
    sids_fids.append("random" + str(sid_num.item())+ "_"+ str(fid.item()))
    
    
# store the data using a dataframe
systems_df = pd.DataFrame({"sids_fids":sids_fids,
                          "sid":sids,
                          "fid":fids,
                          "ref_energies":ref_energies_items,})
systems_df.set_index('sids_fids', inplace = True )
systems_info_dict = systems_df.to_dict('index')

## load tags of atoms mappings {0:subsurface atom, 1: surface atom, 2: adsorbate atom}
with open('../adslab_tags_full.pkl', 'rb') as read_file:
    tags_mappings = pickle.load(read_file)
    
    
sids = []
fids = []
ref_energies = []
surface_anomalies = []
adsorbate_dissociated_anomalies = []
adsorbate_desorbed_anomalies = []
def get_anomaly(sid_fids_split):
    for sid_fid in tqdm(sid_fids_split):
        # extract info from the the dictionary
        sid = systems_info_dict[sid_fid]["sid"]
        # detect anomalies
        init_atoms = ase.io.read(f"/home/jovyan/shared-datasets/OC20/trajs/val_02_01/{sid}.traj", 0)
        final_atoms = ase.io.read(f"/home/jovyan/shared-datasets/OC20/trajs/val_02_01/{sid}.traj", -1)
        tags = tags_mappings[sid]
        anomaly_detector = DetectTrajAnomaly(init_atoms, final_atoms, tags, )
        surface_anomaly = int(anomaly_detector.has_surface_changed())
        adsorbate_dissociated = int(anomaly_detector.is_adsorbate_dissociated())
        adsorbate_desorbed = int(anomaly_detector.is_adsorbate_desorbed())
        ## store the values in a list
        sids.append(sid)
        fids.append(systems_info_dict[sid_fid]["fid"])
        ref_energies.append(systems_info_dict[sid_fid]["ref_energies"])
        surface_anomalies.append(surface_anomaly)
        adsorbate_dissociated_anomalies.append(adsorbate_dissociated)
        adsorbate_desorbed_anomalies.append(adsorbate_desorbed)
    return sids, fids, ref_energies, surface_anomalies, adsorbate_dissociated_anomalies,adsorbate_desorbed_anomalies

## multiprocessing
sid_fids = list(systems_info_dict.keys())
# split up the dictionary that has all the atoms objects
train_trajs_splits = np.array_split(sid_fids, NUM_WORKERS)
# pool over these splits 
pool = Pool(NUM_WORKERS)     
mp_args = [subsplit for subsplit in train_trajs_splits]
all_outputs = list(pool.imap(get_anomaly, mp_args))

# combine the outputs from the pool
for output in all_outputs:
    sids += output[0]
    fids += output[1]
    ref_energies += output[2]
    surface_anomalies += output[3]
    adsorbate_dissociated_anomalies += output[4]
    adsorbate_desorbed_anomalies += output[5]
    
df_30k = pd.DataFrame({"sid": sids,
                        "fid":fids,
                        "ref_energies":ref_energies,
                        "surface_anomalies":surface_anomalies,
                       "adsorbate_dissociated_anomalies":adsorbate_dissociated_anomalies,
                       "adsorbate_desorbed_anomalies":adsorbate_desorbed_anomalies,
                       })

df_30k.to_csv("val_id_30k_anomalies.csv")