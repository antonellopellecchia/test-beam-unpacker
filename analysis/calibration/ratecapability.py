import os, sys
import pathlib
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("ROOT")

saturating_curve = lambda x, A, tau: (A*x) / (1 + tau*(A*x))
ratecapability_curve = lambda rate, tau: 1 / (1 + tau*rate)

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

    """ Read rate files and corresponding x-ray flux: """
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

    """ Fit with saturating curve and plot: """
    vfats = list(enumerate(rate_df["vfat"].unique()))
    vfat_index = dict(vfats)
    vfat_index = { v: k for k, v in vfat_index.items()}

    def analyze_rate(vfat_df):
        """ Plot rate vs source power, fit and calculate efficiency: """

        vfat = vfat_df["vfat"].iloc[0]
        iax = vfat_index[vfat]
        
        flux, rate, err_rate = vfat_df["xray"][1:], vfat_df["rate"][1:]/128, vfat_df["rate_error"][1:]/128
        
        popt = [9e4, 0]
        [a, tau], pcov = curve_fit(saturating_curve, flux, rate, popt, err_rate)
        err_a, err_tau = np.sqrt(np.diag(pcov))
        true_rate, err_true_rate = a*flux, err_a*flux

        efficiency = rate / true_rate
        err_efficiency = efficiency * np.sqrt( err_rate**2/rate**2 + err_true_rate**2/true_rate**2 )

        if args.verbose: print("a={}, tau={}".format(a, tau))

        rate_ax[iax].errorbar(flux, rate/1e6, yerr=err_rate/1e6, fmt=".", color="black")
        rate_ax[iax].plot(
            flux, saturating_curve(flux, a, tau)/1e6, "-",
            color="red", label=f"$\\tau\,=\,{tau*1e9:1.2f}\,\pm{err_tau*1e9:1.2f}\,ns$"
        )
        rate_ax[iax].legend()
        rate_ax[iax].set_xlabel("X-ray current (µA)")
        rate_ax[iax].set_ylabel("Interacting particle rate (MHz/strip)")
        rate_ax[iax].set_title("VFAT {}".format(vfat))

        efficiency_ax[iax].errorbar(true_rate/1e3, efficiency, yerr=err_efficiency, fmt=".", color="black")
        efficiency_ax[iax].plot(
            true_rate/1e3, ratecapability_curve(true_rate, tau), "-",
            color="red", label=f"$\\tau\,=\,{tau*1e9:1.2f}\,\pm{err_tau*1e9:1.2f}\,ns$"
        )
        efficiency_ax[iax].legend()
        efficiency_ax[iax].set_xscale("log")
        efficiency_ax[iax].set_xlabel("Interacting particle rate (kHz/strip)")
        efficiency_ax[iax].set_ylabel("Efficiency")
        efficiency_ax[iax].set_title("VFAT {}".format(vfat))

    nvfats = 4
    nrows = int(nvfats**0.5)
    ncols = int(nvfats/nrows)
    rate_fig, rate_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
    rate_ax = rate_ax.flatten()
    efficiency_fig, efficiency_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
    efficiency_ax = efficiency_ax.flatten()

    rate_df.groupby("vfat").apply(analyze_rate)
    rate_fig.tight_layout()
    efficiency_fig.tight_layout()

    rate_df.to_csv(args.odir / "rate.csv", sep=";")
    if args.verbose: print("Rate csv file saved to", args.odir/"rate.csv")
    rate_fig.savefig(args.odir / "rate.png")
    if args.verbose: print("Rate plots saved to", args.odir/"rate.png")
    efficiency_fig.savefig(args.odir / "efficiency.png")
    if args.verbose: print("Efficiency plots saved to", args.odir/"efficiency.png")

if __name__=="__main__": main()
