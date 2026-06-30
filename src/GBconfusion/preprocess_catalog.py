import argparse
from GBconfusion.preprocess_catalog_tools import preprocess_catalog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LISA binary catalog")

    # Required
    parser.add_argument("--filepath", required=True, help="Path to HDF5 catalog")
    parser.add_argument("--output", required=True, help="Output filename")

    # Optional parameters with defaults
    parser.add_argument("--T_obs", type=float, default=4*365*24*3600, help="Observation time in seconds")
    parser.add_argument("--delta_t", type=float, default=5, help="Time resolution in seconds")
    parser.add_argument("--t0", type=float, default=0, help="Starting time of chunk")
    parser.add_argument("--t_init", type=float, default=0, help="Time of catalog")
    parser.add_argument("--tdi", type=float, default=1.5, help="TDI scaling factor")
    parser.add_argument("--snr_preselection", type=float, default=0.001, help="SNR preselection threshold")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--keys", nargs="+", default=None, help="List of catalog keys to load (space-separated). If not provided, default keys are used.")

    args = parser.parse_args()
    
    preprocess_catalog(args.filepath,
                       args.output,
                       keys=args.keys,
                       T_obs=args.T_obs,
                       delta_t=args.delta_t,
                       tdi=args.tdi,
                       snr_preselection=args.snr_preselection,
                       batch_size=args.batch_size)  

