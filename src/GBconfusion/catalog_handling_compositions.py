import numpy as np
import h5py
from GBconfusion.snr import approx_snr
from GBconfusion.lisa_psd import psd_source_approx
from GBconfusion.lisa_response_fastGB import  tdi_AE_fastGB_multi
from jaxgb.jaxgb import JaxGBaccurate
from jaxgb.params import GBObject
from lisa_response_jaxgb import jaxgb_response_batch
from lisaorbits import EqualArmlengthOrbits
import jax
from scipy.constants import c
import pandas as pd
from tqdm import tqdm
import gc
jax.config.update("jax_enable_x64", True)


def characteristic_strain(T_obs, f0, Amp):
    h_c = (16 /5 * T_obs)**(1/2) * f0 * Amp
    return h_c


def process_catalog_batches(catalog, T_obs, t0, t_init, delta_t, tdi,  batch_size=1000, output_file = 'galactic_binaries_waveforms.hdf5', snr_preselection = 0.001, verbose=True):
    """
    Function to process the catalog in batches

    Parameters:
    ------------------
    catalog: contains binaries parameters
    T_obs: in s
    t0: starting time 
    t_init: time of the catalog
    delta_t: sampling time in s
    batch_size: 
    output_file: name of output file
    snr_preselection: threshold for pre-exclusion of sources to avoid calculating waveform
    ------------------
    Returns: output file with waveforms, psd estimate, position of source, source's properties, observational parameters
    """

    n_tot = len(catalog["Amplitude"])
    n_batches = int(np.ceil(n_tot/batch_size))

    if verbose:
        print(f"Processing {n_tot} sources")
        print(f"Batches: {n_batches} batches with {batch_size} sources")
    
    # Create different N values to assign depending on the required N (depends on the fdot)
    N_values = {
    "small": 128,
    "medium": 512,
    "large": 2048
    }
    bucket_items = list(N_values.items())

    with h5py.File(output_file, 'w') as f:

        # This creates a dataset for all sources, where I save the waveforms and useful parameters of each batch. Save them in buckets of different Ns to save space
        # do NOT store individual PSDs 
        meta = f.create_group('meta')
        meta_f0 = meta.create_dataset('f0', shape=(n_tot,), dtype='float64')
        meta_fdot = meta.create_dataset('fdot', shape=(n_tot,), dtype='float64')
        meta_ampl = meta.create_dataset('Ampl', shape=(n_tot,), dtype='float64')
        meta_dist = meta.create_dataset('lum_dist', shape=(n_tot,), dtype='float64')
        
        psd_est = f.create_dataset('source_psd_estimate', shape=(n_tot,), dtype='float64')

        for bname, N in bucket_items:
            grp = f.create_group(bname)
            grp.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int') # to keep track of the source in the whole catalog
            grp.create_dataset('A', shape=(0, N), maxshape=(None, N), dtype='complex128')
            grp.create_dataset('E', shape=(0, N), maxshape=(None, N), dtype='complex128')
            grp.create_dataset('fr', shape=(0, N), maxshape=(None, N), dtype='float64')
            grp.create_dataset('Dec', shape=(0,), maxshape=(None,), dtype='float64')
            grp.create_dataset('RA', shape=(0,), maxshape=(None,), dtype='float64')
    
        # Store observational parameters as attributes
        f.attrs['T_obs'] = T_obs
        f.attrs['N_values'] = list(N_values.values())
        f.attrs['delta_t'] = delta_t

        # Process the batch
        for i in tqdm(range(n_batches), desc="Processing batches", disable=not verbose, mininterval=1.0):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_tot)
            current_batch_size = end_idx - start_idx
            
            # Extract this batch from the full catalog

            batch_params = pd.DataFrame({
                catalog['Frequency'][start_idx:end_idx][:, 0],
                catalog['FrequencyDerivative'][start_idx:end_idx][:, 0],
                catalog['Amplitude'][start_idx:end_idx][:, 0],
                catalog['EquatorialLatitude'][start_idx:end_idx][:, 0],
                catalog['EquatorialLongitude'][start_idx:end_idx][:, 0],
                catalog['Polarization'][start_idx:end_idx][:, 0],
                catalog['Inclination'][start_idx:end_idx][:, 0],
                catalog['InitialPhase'][start_idx:end_idx][:, 0]
                
            })
            
            meta_f0[start_idx:end_idx] = batch_params[:, 0]
            meta_fdot[start_idx:end_idx] = batch_params[:, 1]
            meta_ampl[start_idx:end_idx] = batch_params[:, 2]
            meta_dist[start_idx:end_idx] = catalog['LuminosityDistance'][start_idx:end_idx][:, 0]
            
            # Rough estimation of the SNR of sources to avoid computing the waveform of already weak sources
            # For weak sources, compute the estimated PSD and store it (it will contribute to the background)
            fdot = batch_params[:, 1]
            f0 = batch_params[:,0]
            amp = batch_params[:,2]

            h_c = characteristic_strain(T_obs, f0, amp)
            SNR_approx = approx_snr(h_c, f0)
            psd_est[start_idx:end_idx] = psd_source_approx(h_c, f0, tdi)

            # Mask the loud sources (possibly resolvable) using a SNR threshold. Skipped sources have None instead of the waveform
            loud_sources_mask = SNR_approx > snr_preselection
                
            # Assign N based on fdot to compute the waveform with FastGB for the loud sources
            required_bins = np.abs(fdot) * T_obs**2  

            bucket = np.full(len(f0),'skip', dtype='U10') # initialize the bukcets with all skip

            bucket[(required_bins < 1e2) & loud_sources_mask] = "small"
            bucket[(required_bins >= 1e2) & (required_bins < 1e3) & loud_sources_mask] = "medium"
            bucket[(required_bins >= 1e3) & loud_sources_mask] = "large"

            # process each bucket individually
            for bname, N in N_values.items():
                mask = (bucket == bname)
                if not np.any(mask):
                    continue

                params_sub = batch_params[mask]
                gbo     = GBObject.from_pandas_dataframe(params_sub, t_init=t_init)
                params_sub_jx  = gbo.to_jaxgb_array(t0=t0)

                idxs = mask.nonzero()[0]
                global_idxs = start_idx + idxs

                orbits = EqualArmlengthOrbits()
                myjaxgb = JaxGBaccurate(window=0.5, orbits=orbits, t_obs=T_obs, t0=t0, n=N)
                tdi_fn  = jax.jit(lambda p: myjaxgb.get_tdi(p, tdi_generation=2.0, tdi_combination="AET"))

                A_sub, E_sub,  fr_sub = jaxgb_response_batch(
                    params = params_sub_jx,
                    t_obs = T_obs,
                    n=N,
                    myjaxgb=myjaxgb,
                    tdi_fn=tdi_fn
                )

                # Save position for sources with waveform
                ecliptic_lat_sub = params_sub[:,3] 
                ecliptic_lon_sub = params_sub[:,4]

                grp = f[bname]
                old = grp['indices'].shape[0]
                new = old + len(global_idxs)
                
                for dset in ('indices', 'Dec', 'RA'):
                    grp[dset].resize((new,))
                for dset in ('A', 'E', 'fr'):
                    grp[dset].resize((new, N))
            

                grp['indices'][old:new] = global_idxs
                grp['A'][old:new] = A_sub
                grp['E'][old:new] = E_sub
                grp['fr'][old:new] = fr_sub
                grp['Dec'][old:new] = ecliptic_lat_sub  
                grp['RA'][old:new] = ecliptic_lon_sub

                # Clear memory
                del A_sub, E_sub, fr_sub, ecliptic_lat_sub, ecliptic_lon_sub
            if i % 1000 == 0: 
                gc.collect()
        if verbose:
            print(f"Saved {n_tot} waveforms sources to {output_file}")
            print(f"File size: {f.id.get_filesize() / (1024**3):.2f} GB")
        
    return output_file
