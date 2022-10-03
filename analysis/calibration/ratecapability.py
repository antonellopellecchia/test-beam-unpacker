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

import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", OptimizeWarning)

saturating_curve = lambda x, A, tau: (A*x) / (1 + tau*(A*x))
ratecapability_curve = lambda rate, tau: 1 / (1 + tau*rate)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("rundir", type=pathlib.Path)
    #parser.add_argument("run_number", type=str)
    parser.add_argument("odir", type=pathlib.Path)
    parser.add_argument("--strips", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pulse-stretch", type=int, default=7) 
    parser.add_argument("-n", "--events", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)

    """ Read rate files and corresponding x-ray flux: """
    run_dir = args.rundir #pathlib.Path("/home/gempro/testbeam/july2022/runs/rate-xray/daq-monitor")
    #run_number = args.run_number
    #run_dir = rate_dir / f"{run_number}/rates"
    #if args.strips:
    #    run_dir = rate_dir / f"rates/{run_number}" # use path for DAQ data
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

    deadtime_tuples = list()
    def analyze_rate(vfat_df):
        """ Plot rate vs source power, fit and calculate efficiency: """

        vfat = vfat_df["vfat"].iloc[0]
        iax = vfat_index[vfat]
      
        def plot_and_fit_rate(df, result_type="broadcast"):
            
            flux, rate, err_rate = df["xray"][1:], df["rate"][1:], df["rate_error"][1:]

            popt = [9e4, 0]
            [slope, tau], pcov = curve_fit(saturating_curve, flux, rate, popt, err_rate)
            if tau <= 0 or np.isinf(pcov).any(): return # skip channel

            err_slope, err_tau = np.sqrt(np.diag(pcov))
            true_rate, err_true_rate = slope*flux, err_slope*flux

            efficiency = rate / true_rate
            err_efficiency = efficiency * np.sqrt( err_rate**2/rate**2 + err_true_rate**2/true_rate**2 )

            if args.verbose and not args.strips: print("slope = {}, tau = {}".format(slope, tau))

            if args.strips:
                # do not plot efficiency and save deadtime
                channel = df["channel"].iloc[0]
                deadtime_tuples.append((vfat, channel, tau))
                return pd.DataFrame({
                    "vfat": vfat,
                    "channel": channel,
                    "slope": slope,
                    "tau": tau,
                    "rate": true_rate,
                    "efficiency": efficiency
                })
            
            rate_ax.flat[iax].errorbar(flux, rate/1e6, yerr=err_rate/1e6, fmt=".", color="black")
            rate_ax.flat[iax].plot(
                flux, saturating_curve(flux, slope, tau)/1e6, "-",
                color="red", label=f"$\\tau\,=\,{tau*1e9:1.2f}\,\pm {err_tau*1e9:1.2f}\,ns$"
            )

            efficiency_ax.flat[iax].errorbar(true_rate/1e6, efficiency, yerr=err_efficiency, fmt=".", color="black")
            efficiency_ax.flat[iax].plot(
                true_rate/1e6, ratecapability_curve(true_rate, tau), "-",
                color="red", label=f"$\\tau\,=\,{tau*1e9:1.2f}\,\pm {err_tau*1e9:1.2f}\,ns$"
            )
                
        if not args.strips:
            plot_and_fit_rate(vfat_df)
        else:
            efficiency_df = vfat_df.groupby("channel").apply(plot_and_fit_rate)

            if not efficiency_df.empty and args.strips:
                print(efficiency_df)
                taus, slopes = efficiency_df["tau"]*1e9, efficiency_df["slope"]
                tau_filter = (np.abs(taus-taus.mean())<2*taus.std()) & (np.abs(slopes-slopes.mean())<2*slopes.std())
                taus, slopes = taus[tau_filter], slopes[tau_filter]
                tau_range = (0.5*taus.min(), 2*taus.max())
                #tau_range = (0, 60e3)
                tau_ax.flat[iax].hist(taus, bins=20, range=tau_range, alpha=0.5)
                tau_ax.flat[iax].set_xlabel("Time constant (ns)")
                tau_ax.flat[iax].set_title("VFAT {}".format(vfat))
            
                slope_ax.flat[iax].hist(slopes, bins=20, alpha=0.5)
                slope_ax.flat[iax].set_xlabel("Slope (kHz/µA)")
                slope_ax.flat[iax].set_title("VFAT {}".format(vfat))

                tau_slope_ax.flat[iax].plot(slopes, taus, ".")
                tau_slope_ax.flat[iax].set_xlabel("Slope (kHz/µA)")
                tau_slope_ax.flat[iax].set_ylabel("Time constant (ns)")
                tau_slope_ax.flat[iax].set_title("VFAT {}".format(vfat))
           
            cmap_new = mpl.cm.get_cmap("viridis")
            cmap_new.set_under("w")
            my_norm = mpl.colors.Normalize(vmin=.25, vmax=5, clip=False)
            print("Default point size", mpl.rcParams['lines.markersize'] ** 2)
            rate_img = rate_ax.flat[iax].scatter(
                vfat_df["channel"], vfat_df["xray"], c=vfat_df["rate"],
                cmap=cmap_new, s=1000#, norm=my_norm
            )
            plt.colorbar(rate_img)

            efficiency_ax.flat[iax].scatter(
                efficiency_df["channel"], efficiency_df["rate"], c=efficiency_df["efficiency"]
            )

        if not args.strips: rate_ax.flat[iax].legend()
        rate_ax.flat[iax].set_xlabel("X-ray current (µA)")
        rate_ax.flat[iax].set_ylabel("Interacting particle rate (MHz)")
        rate_ax.flat[iax].set_title("VFAT {}".format(vfat))

        if not args.strips:
            efficiency_ax.flat[iax].legend()
            efficiency_ax.flat[iax].set_xscale("log")
            efficiency_ax.flat[iax].set_xlabel("Interacting particle rate (MHz)")
            efficiency_ax.flat[iax].set_ylabel("Efficiency")
        else:
            efficiency_ax.flat[iax].set_xlabel("Channel")
            efficiency_ax.flat[iax].set_ylabel("Rate")

            rate_ax.flat[iax].set_xlabel("Channel")
            rate_ax.flat[iax].set_ylabel("X-ray current (µA)")
            #plt.colorbar()
        efficiency_ax.flat[iax].set_title("VFAT {}".format(vfat))

    nvfats = 4
    nrows = int(nvfats**0.5)
    ncols = int(nvfats/nrows)
    rate_fig, rate_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
    efficiency_fig, efficiency_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
    
    if args.strips:
        tau_fig, tau_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
        slope_fig, slope_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
        tau_slope_fig, tau_slope_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))

    rate_df.groupby("vfat").apply(analyze_rate)
    rate_fig.tight_layout()
    efficiency_fig.tight_layout()
    if args.strips: tau_fig.tight_layout()

    rate_df.to_csv(args.odir / "rate.csv", sep=";")
    if args.verbose: print("Rate csv file saved to", args.odir/"rate.csv")
    rate_fig.savefig(args.odir / "rate.png")
    if args.verbose: print("Rate plots saved to", args.odir/"rate.png")
    efficiency_fig.savefig(args.odir / "efficiency.png")
    if args.verbose: print("Efficiency plots saved to", args.odir/"efficiency.png")

    if args.strips:
        deadtime_df = pd.DataFrame(deadtime_tuples, columns=["vfat", "channel", "tau"])
        deadtime_df.to_csv(args.odir / "deadtime.csv", sep=";")
        tau_fig.savefig(args.odir / "tau.png")
        slope_fig.savefig(args.odir / "slope.png")
        tau_slope_fig.savefig(args.odir / "tau_slope.png")

if __name__=="__main__": main()
