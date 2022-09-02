import os, sys
import pathlib
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("ROOT")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("rundir", type=pathlib.Path)
    parser.add_argument("run_number", type=str)
    parser.add_argument("odir", type=pathlib.Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pulse-stretch", type=int, default=7) 
    parser.add_argument("-n", "--events", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)

    rate_dir = args.rundir #pathlib.Path("/home/gempro/testbeam/july2022/runs/rate-xray/daq-monitor")
    run_number = args.run_number
    run_dir = rate_dir / f"{run_number}/rates"
    rate_files = os.listdir(run_dir)
    if args.verbose: print(rate_files)

    def read_rate_df(rate_dir, f):
        current = int(f.split(".")[0])
        rate_df = pd.read_csv(rate_dir / f, sep=";")
        rate_df["xray"] = current
        return rate_df

    rate_df = pd.concat([
        read_rate_df(run_dir, f)
        for f in rate_files
    ])
    if args.verbose:
        print("Rate dataframe:")
        print(rate_df)

    saturating_curve = lambda x, A, tau: (A*x) / (1 + tau*(A*x))

    vfats = list(enumerate(rate_df["vfat"].unique()))
    vfat_index = dict(vfats)
    vfat_index = { v: k for k, v in vfat_index.items()}

    def plot_rate(vfat_df):
        vfat = vfat_df["vfat"].iloc[0]
        iax = vfat_index[vfat]
        
        flux, rate, err_rate = vfat_df["xray"][1:], vfat_df["rate"][1:]/128, vfat_df["rate_error"][1:]/128
        
        popt = [9e4, 0]
        [a, tau], pcov = curve_fit(saturating_curve, flux, rate, popt, err_rate)
        err_a, err_tau = np.sqrt(np.diag(pcov))

        if args.verbose: print("a={}, tau={}".format(a, tau))

        ax[iax].errorbar(flux, rate/1e6, yerr=err_rate/1e6, fmt=".", color="black")
        ax[iax].plot(
            flux, saturating_curve(flux, a, tau)/1e6, "-",
            color="red", label=f"$\\tau\,=\,{tau*1e9:1.2f}\,\pm{err_tau*1e9:1.2f}\,ns$"
        )
        ax[iax].legend()
        ax[iax].set_xlabel("X-ray current (µA)")
        ax[iax].set_title("VFAT {}".format(vfat))
        ax[iax].set_ylabel("Interacting particle rate (MHz/strip)")
            
    nvfats = 4
    nrows = int(nvfats**0.5)
    ncols = int(nvfats/nrows)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
    ax = ax.flatten()

    rate_df.groupby("vfat").apply(plot_rate)
    fig.tight_layout()

    rate_df.to_csv(args.odir / "rate.csv", sep=";")
    fig.savefig(args.odir / "rate.png")

if __name__=="__main__": main()
