from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Outcar
import ase
import os

rootdir_glob = '/home/jovyan/shared-scratch/kabdelma/OC20_dataset_errors/single_pts_dft/experiments_uuids/'


def unconverged_scf(df):
    uuids = df.index.values
    results = {}
    for uuid in tqdm(uuids):
        max_nelm = df.loc[uuid]["max_nelm"]
        nelm = df.loc[uuid]["nelm"]
        last_scf_cycle_dE = df.loc[uuid]["last_scf_cycle_dE"]
        if (np.abs(last_scf_cycle_dE)>0.0001) & (nelm==max_nelm):
            results[uuid] = last_scf_cycle_dE
    return results


def energy_encut_conv(df,encut_values, material):
    all_final_energies = []
    for sid in tqdm(np.unique(df.oc20_sid)):
        df_sid = df.query(f"oc20_sid == '{sid}'") 
        final_energies = [] 
        for encut in encut_values: 
            final_energies.append(df_sid.query(f"encut=={encut}")['final_energy'].values[0])
        all_final_energies.append(final_energies)
        plt.plot(encut_values, np.array(final_energies) - final_energies[-1], "-o",) 

    plt.title(f"{material} energy convergence vs encut")       
    plt.xlabel("encut (ev)")
    plt.ylabel("Energy convergence error (ev)");
#     plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left");
    return np.array(all_final_energies)



def force_encut_conv(df,encut_values, material):
    for sid in tqdm(np.unique(df.oc20_sid)):
            df_sid = df.query(f"oc20_sid == '{sid}'") 
            all_final_forces = []
            final_forces = [] 
            for encut in encut_values: 
                final_forces.append(df_sid.query(f"encut=={encut}")['unconstrained_atoms_forces'].values[0])
            all_final_forces.append(final_forces)
            plt.plot(encut_values, 
                     np.linalg.norm(np.array(final_forces) - final_forces[-1], axis = (1,2)), "-o",) 
    plt.title(f"{material} force convergence vs encut")       
    plt.xlabel("encut (ev)")
    plt.ylabel("force convergence error (ev/Angst)");
#     plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left");
    return np.array(all_final_forces)

def slab_adslab_encut_Econv(slabs_df, adslabs_df, encut_values, material):
    ## (adslab - slab) convergence results
    all_energy_differences = []
    for sid in tqdm(np.unique(adslabs_df.oc20_sid)):
        try:
            slab_df = slabs_df.query(f"updated_sid == '{sid}'")
            adslab_df = adslabs_df.query(f"oc20_sid == '{sid}'") 
            energy_differences = []
            for encut in encut_values : 
                # get the energy of the slab 
                slab_final_energy = slab_df.query(f"encut=={encut}")['final_energy'].values[0]
                ## get the energy of the adslab
                adslab_final_energy = adslab_df.query(f"encut=={encut}")['final_energy'].values[0]
                energy_differences.append(adslab_final_energy - slab_final_energy)
            all_energy_differences.append(energy_differences)
            plt.plot(encut_values, 
                     np.array(energy_differences) - energy_differences[-1], "-o",)
        except IndexError as exception:
            pass
    plt.title(f"{material} (adslab - slab) convergence vs encut")       
    plt.xlabel("encut (ev)")
    plt.ylabel("Energy convergence error (ev)");
    return np.array(all_energy_differences)


def sigma_Econv(df, sigma_values, material):
    all_final_energies = []
    for sid in tqdm(np.unique(df.oc20_sid)):
        try: 
            df_sid = df.query(f"oc20_sid == '{sid}'") 
            final_energies = [] 
            for sig in sigma_values : 
                final_energies.append(df_sid.query(f"sigma=={sig}")['final_energy'].values[0])
            all_final_energies.append(final_energies)
            plt.plot(sigma_values, 
                     np.array(final_energies) - final_energies[-1], "-o",) 
        except IndexError as exception:
            pass
    plt.title(f"{material} convergence vs sigma")       
    plt.xlabel("sigma")
    plt.ylabel("Energy convergence error (ev)");
    return np.array(all_final_energies)


def kpt_convergence(df,kpts_values, material):
    for sid in tqdm(np.unique(df.oc20_sid)):
            df_sid = df.query(f"oc20_sid == '{sid}'") 
            final_energies = [] 
            for kpt in kpts_values: 
                final_energies.append(df_sid.query(f"k_point_multiplier=={kpt}")['final_energy'].values[0])
            plt.plot(kpts_values, 
                     np.array(final_energies) - final_energies[-1], "-o",)
    plt.title(f"{material} convergence vs k-points")       
    plt.xlabel("K-point multiplier")
    plt.ylabel("Energy convergence error (ev)");