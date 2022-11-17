import os, sys
import argparse
import pathlib

import numpy as np
import pandas as pd
import awkward as ak
import uproot

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", type=pathlib.Path)
    parser.add_argument("odir", type=pathlib.Path)
    parser.add_argument("method", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--rundir", type=pathlib.Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pulse-stretch", type=int, default=0) 
    parser.add_argument("-n", "--events", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)

    if args.method == "daq":
        
        print("Pulse stretch", args.pulse_stretch)

        with uproot.open(args.ifile) as input_file:
            input_tree = input_file["outputtree"]

            print("Reading input tree...")
            chambers = input_tree["digiChamber"].array(entry_start=args.start, entry_stop=args.events)
            etas = input_tree["digiEta"].array(entry_start=args.start, entry_stop=args.events)
            strips = input_tree["digiStrip"].array(entry_start=args.start, entry_stop=args.events)
        
            rate_tuples = list()

            total_triggers = ak.num(chambers, axis=0)
            daq_time = total_triggers * (args.pulse_stretch+1) * 25e-9
            print(f"Taken {total_triggers} triggers, DAQ time {daq_time*1e9:1.0f} ns")
            
            chambers_unique = np.unique(ak.flatten(chambers))
            etas_unique = np.unique(ak.flatten(etas))
            nrows, ncols = len(etas_unique), len(chambers_unique)
            rate_fig, rate_axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols, 9*nrows))
            rate2d_fig, rate2d_axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols, 9*nrows))

            for ichamber,chamber in enumerate(np.unique(ak.flatten(chambers))):

                eta_chamber = etas[chambers==chamber]
                strips_chamber = strips[chambers==chamber]

                for ieta,eta in enumerate(np.unique(ak.flatten(eta_chamber))):

                    strips_eta = strips_chamber[eta_chamber==eta]
                    strips_unique = np.unique(ak.flatten(strips_eta))

                    min_strip, max_strip = ak.min(strips_eta, axis=None), ak.max(strips_eta, axis=None)
                    strip_bins = max_strip - min_strip + 1
                    strip_range = (min_strip-0.5, max_strip+0.5)
                    rate_axs[ieta][ichamber].hist(
                        ak.flatten(strips_eta),
                        weights=np.ones(len(ak.flatten(strips_eta)))/daq_time/1e3,
                        bins=strip_bins, range=strip_range,
                        histtype="step", color="blue", linewidth=2
                    )
                    rate_axs[ieta][ichamber].set_xlabel("Strip")
                    rate_axs[ieta][ichamber].set_ylabel("Rate (kHz)")
                    rate_axs[ieta][ichamber].set_title("Detector {} eta {}".format(chamber, eta))

                    for strip in strips_unique:
                        strip_occurrency = ak.count_nonzero(ak.flatten(strips_eta==strip))
                        event_rate = strip_occurrency / daq_time
                        event_rate_error = np.sqrt(strip_occurrency) / daq_time
                        if args.verbose: print("chamber {}, eta {}, strip {}, count {}; rate {}".format(chamber, eta, strip, strip_occurrency, event_rate))
                        rate_tuples.append((chamber, eta, strip, strip_occurrency, total_triggers, event_rate, event_rate_error))
                        
            rate_df = pd.DataFrame(rate_tuples, columns=["chamber", "eta", "strip", "counts", "triggers", "rate", "rate_error"])
            rate_df.to_csv(args.odir / "rate.csv", index=None)
            print("Rates per strip saved to", args.odir / "rate.csv")

            rate_fig.tight_layout()
            rate_fig.savefig(args.odir / "rate.png")
            rate_fig.savefig(args.odir / "rate.pdf")

            rate_avg_df = rate_df.groupby(["chamber", "eta"]).apply(np.mean)
            print("Average rates:")
            print(rate_avg_df)
            rate_avg_df.to_csv(args.odir / "rate_average.csv", index=None)
            print("Average rates saved to", args.odir / "rate_average.csv")

    elif args.method == "sbit":

        rate_df = pd.read_csv(args.ifile, sep=";")
        rate_df["slot"] = 0

        print("Rate measurement with sbit discontinued for now. Please revert to a previous commit if you really need this.")
        sys.exit(1)

    elif args.method == "scan":

        runs_df = pd.read_csv(args.ifile)
        print(runs_df)

        rate_dataframes = list()
        for run_number in runs_df.run:
            source_abs = runs_df[runs_df.run==run_number].attenuation.iloc[0]
            run_csv = args.rundir / f"{run_number:04d}/rate_average.csv"
            run_df = pd.read_csv(run_csv)
            run_df["source_abs"] = source_abs
            rate_dataframes.append(run_df)
        rate_df = pd.concat(rate_dataframes)[["source_abs", "chamber", "eta", "rate"]]

        rates_average_df = rate_df.groupby(["chamber", "source_abs"]).apply(np.mean)

        chambers = rates_average_df.chamber.unique()
        rate_fig, rate_axs = plt.subplots(nrows=1, ncols=len(chambers), figsize=(11*len(chambers), 9))
        for ichamber,chamber in enumerate(chambers):
            ax = rate_axs[ichamber]
            chamber_rate_df = rates_average_df[rates_average_df.chamber==chamber]
            ax.plot(
                chamber_rate_df.source_abs, chamber_rate_df.rate,
                "o", color="black", label=f"Detector {chamber:1.0f}"
            )
            ax.legend()
            ax.set_xlabel("Source absorption factor")
            ax.set_ylabel("Average rate (kHz/strip)")
            hep.cms.text("Muon Preliminary", ax=ax)
            ax.set_xscale("log")
            ax.set_yscale("log")

        rates_average_df.to_csv(args.odir / "rate.csv")
        print("Rate file saved to", args.odir / "rate.csv")

        rate_fig.tight_layout()
        rate_fig.savefig(args.odir / "rate.png")
        rate_fig.savefig(args.odir / "rate.pdf")

if __name__=='__main__': main()
