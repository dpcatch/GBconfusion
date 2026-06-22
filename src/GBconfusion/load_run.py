
import numpy as np
import h5py
import pandas as pd
from GBconfusion.iteration_utils import _build_run_output

def load_run(filepath):
    """Load a previously saved run from an HDF5 file."""
    with h5py.File(filepath, "r") as f:
        # reconstruct the raw results/state shapes that _build_run_output expects
        resolved_sources = []
        for key in sorted(f["resolved_sources"].keys()):
            g = f["resolved_sources"][key]
            src = {
                "id":          g.attrs["id"],
                "f0":          g.attrs["f0"],
                "fdot":        g.attrs["fdot"],
                "Ampl":        g.attrs["Ampl"],
                "ecliptic_lat": g.attrs.get("ecliptic_lat", np.nan),
                "ecliptic_lon": g.attrs.get("ecliptic_lon", np.nan),
                "lum_dist":    g.attrs.get("lum_dist", np.nan),
                "A":           g["A"][:],
                "E":           g["E"][:],
                "fr":          g["fr"][:],
            }
            resolved_sources.append({"source": src, "snr": g.attrs["snr"]})

        psd_confusion = [
            (int(key.split("_")[1]), f["psd_confusion"][key]["psd_total"][:])
            for key in sorted(f["psd_confusion"].keys())
        ]

        hist_grp = f["history"]
        n = len(hist_grp["iteration"])
        history = [
            {name: hist_grp[name][i] for name in hist_grp.keys()}
            for i in range(n)
        ]

        results = {
            "resolved_sources":       resolved_sources,
            "global_fr":              f["global_fr"][:],
            "psd_confusion":          psd_confusion,
            "resolved_global_indices": f["resolved_global_indices"][:],
            "n_resolved":             len(resolved_sources),
            "iterations":             f.attrs["iterations"],
            "history":                history,
        }

        # minimal state reconstruction for _build_run_output
        state = {
            "T_obs":        f.attrs["T_obs"],
            "snr_threshold": f.attrs["snr_threshold"],
            "waveforms":    {"f0": np.zeros(f.attrs["n_total_sources"])},
        }

    return _build_run_output(results, state)