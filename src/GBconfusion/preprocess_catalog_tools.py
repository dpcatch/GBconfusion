# preprocess_catalog.py
import h5py
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
from GBconfusion.catalog_handling_compositions import process_catalog_batches
import gc
import argparse

def preprocess_catalog(filepath, output_filename,
                       keys=None, T_obs=4*365*24*3600,
                       delta_t=5, t0=0, t_init=0, tdi=1.5, snr_preselection=0.001,
                       batch_size=1000):
    """
    Load the catalog, convert coordinates, and process waveforms.
    Saves the waveforms to `output_filename`.
    """
    print("Loading catalog...")
    if keys is None:
        keys = ["Frequency", "FrequencyDerivative",
                "Amplitude", "EquatorialLatitude", "EquatorialLongitude", "Polarization",
                "InitialPhase",  "Inclination", "LuminosityDistance"] #, "SecondaryMassSSBFrame", "PrimaryMassSSBFrame", "TotalMassSSBFrame"]

    # Load catalog
    with h5py.File(filepath, "r") as f:
        param_binaries = {name: f[name][:] for name in keys}

    # Coordinate conversion
    ra = param_binaries.pop("EquatorialLongitude")
    dec = param_binaries.pop("EquatorialLatitude")
    param_binaries["RightAscension"] = ra
    param_binaries["Declination"] = dec

    # Clean memory
    del c, ecl, ra, dec
    gc.collect()

    print("Processing catalog...")
    # Process catalog in batches batches
    process_catalog_batches(param_binaries,
                            T_obs=T_obs,
                            t0 =t0,
                            t_init = t_init,
                            delta_t=delta_t,
                            tdi=tdi,
                            batch_size=batch_size,
                            snr_preselection=snr_preselection,
                            output_file=output_filename,
                            verbose=True)

    # Clean memory
    del param_binaries
    gc.collect()
    print(f"Catalog pre-processing done. Output saved to {output_filename}")

