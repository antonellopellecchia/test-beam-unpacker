import os, sys, pathlib
import argparse

import uproot
import numpy as np
import pandas as pd
import awkward as ak
import scipy
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rundir", type=pathlib.Path, help="Run input directory")
    parser.add_argument('odir', type=pathlib.Path, help="Output directory")
    parser.add_argument("--hv", type=pathlib.Path, help="csv file containing hv mapping for each point")
    parser.add_argument("--chambers", type=int, nargs="+", help="Chambers to plot")
    parser.add_argument("--labels", type=str, nargs="+", help="Chamber labels")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate logging")
    args = parser.parse_args()
    
    os.makedirs(args.odir, exist_ok=True)

    hv_df = pd.read_csv(args.hv, sep=";")
    if args.verbose: print("Runs:\n", hv_df)

    efficiency_dfs = list()
    for irun,row in hv_df.iterrows():
        run, hv = row[["run", "hv"]]
        run_file = args.rundir / f"{run:08d}_chamber.csv"
        if args.verbose: print("Opening efficiency file", run_file)

        run_df = pd.read_csv(run_file, sep=";")
        run_df["hv"] = hv
        run_df["run"] = run

        efficiency_dfs.append(run_df)

    efficiency_df = pd.concat(efficiency_dfs)

    efficiency_fig, efficiency_ax = plt.figure(figsize=(12,9)), plt.axes()
    for chamber, label in zip(args.chambers, args.labels):
    #for chamber in efficiency_df["chamber"].unique():
        chamber_df = efficiency_df[efficiency_df["chamber"]==chamber].sort_values(by="hv")
        print(chamber_df)
        efficiency_ax.plot(chamber_df["hv"], chamber_df["efficiency"], "o-", label=label)
    efficiency_ax.legend()
    efficiency_ax.set_xlabel("Equivalent divider current (µA)")
    efficiency_ax.set_ylabel("Efficiency")
    efficiency_fig.savefig(args.odir / "efficiency.png")
    efficiency_fig.savefig(args.odir / "efficiency.pdf")

if __name__=='__main__': main()
