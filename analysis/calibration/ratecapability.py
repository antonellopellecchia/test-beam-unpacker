import os, sys
import pathlib
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import mplhep
mplhep.set_style("ROOT")

import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", OptimizeWarning)

saturating_curve = lambda x, A, tau, offset: (A*x + offset) / (1 + tau*(A*x + offset))
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
            
            channel = df["channel"].iloc[0]
            flux, rate, err_rate = df["xray"][1:], df["rate"][1:], df["rate_error"][1:]

            popt = [9e4, 0, 0]
            try:
                [slope, tau, offset], pcov = curve_fit(saturating_curve, flux, rate, popt, err_rate)
                err_slope, err_tau, err_offset = np.sqrt(np.diag(pcov))
                if tau<0 or np.isinf(pcov).any(): return # skip channel
            except RuntimeError:
                print("Could not fit, skipping VFAT {}, channel {}".format(vfat, channel))
                slope, tau, offset = popt
                err_slope, err_tau, err_offset = [0, 0, 0] 

            if args.verbose and not args.strips: print("slope = {}, tau = {}".format(slope, tau))

            #print("Flux:", list(flux))
            #print("Rate:", list(rate))
            rate_ax.flat[iax].errorbar(flux, rate/1e6, yerr=err_rate/1e6, fmt=".")#, color="black")
            rate_ax.flat[iax].plot(
                flux, saturating_curve(flux, slope, tau, offset)/1e6, "-",
                label=f"$\\tau\,=\,{tau*1e9:1.2f}\,\pm {err_tau*1e9:1.2f}\,ns$",
                color="red",#color = plt.cm.hot(channel/128)
            )

            if args.strips:
                # do not plot efficiency and save deadtime
                channel = df["channel"].iloc[0]
                deadtime_tuples.append((vfat, channel, tau, rate.mean()))
                return pd.Series({
                    "vfat": vfat,
                    "channel": channel,
                    "slope": slope,
                    "tau": tau,
                    "err_tau": err_tau,
                    "min_rate": rate.min()
                    #"rate": true_rate,
                    #"efficiency": efficiency
                })

                            
        if not args.strips:
            plot_and_fit_rate(vfat_df)
        else:
            efficiency_df = vfat_df.groupby("channel").apply(plot_and_fit_rate)

            avg_rate = vfat_df.groupby("xray").apply(lambda df: df["rate"].mean())
            err_avg_rate = vfat_df.groupby("xray").apply(lambda df: df["rate"].std()/len(df["rate"])**0.5)
            flux = vfat_df["xray"].unique()
            try:
                popt = [9e4, 0, 0]
                [slope, tau, offset], pcov = curve_fit(saturating_curve, flux, avg_rate, popt, err_avg_rate)
                err_slope, err_tau, err_offset = np.sqrt(np.diag(pcov))
                if tau<0 or np.isinf(pcov).any(): return # skip channel
                avg_rate_ax.flat[iax].errorbar(flux, avg_rate/1e6, yerr=err_avg_rate/1e6, fmt=".")#, color="black")
                avg_rate_ax.flat[iax].set_xlabel("X-ray current (µA)")
                avg_rate_ax.flat[iax].set_ylabel("Average particle rate (MHz/strip)")
                avg_rate_ax.flat[iax].set_title("VFAT {}".format(vfat))
                avg_rate_ax.flat[iax].plot(
                    flux, saturating_curve(flux, slope, tau, offset)/1e6, "-",
                    label="measured rate",
                    color="red",#color = plt.cm.hot(channel/128)
                )
                avg_rate_ax.flat[iax].plot(
                    flux, (flux*slope+offset)/1e6, "--",
                    label="true rate",
                    color="blue",#color = plt.cm.hot(channel/128)
                )
                avg_rate_ax.flat[iax].legend()
                avg_rate_ax.flat[iax].text(
                    .9, .1, f"$\\tau$ = {tau*1e9:1.2f} $\pm$ {err_tau*1e9:1.2f} ns",
                    ha = "right", va="center",
                    transform = avg_rate_ax.flat[iax].transAxes
                )
 
                true_rate, err_true_rate = slope*flux + offset, err_slope*flux
                efficiency = avg_rate / true_rate
                err_efficiency = efficiency * np.sqrt( err_avg_rate**2/avg_rate**2 + err_true_rate**2/true_rate**2 )

                efficiency_ax.flat[iax].errorbar(true_rate/1e3, efficiency, yerr=err_efficiency, fmt=".", color="black")
                efficiency_ax.flat[iax].set_xscale("log")
            except RuntimeError:
                print("Could not fit, skipping average plot for VFAT {}, channel {}".format(vfat, channel))
            
            
            if not efficiency_df.empty and args.strips:
                print(efficiency_df)
                taus, slopes, err_taus = efficiency_df["tau"]*1e9, efficiency_df["slope"], efficiency_df["err_tau"]*1e9
                tau_filter = (np.abs(taus-taus.mean())<2*taus.std()) & (np.abs(slopes-slopes.mean())<2*slopes.std())
                taus, slopes, err_taus = taus[tau_filter], slopes[tau_filter], err_taus[tau_filter]
                if len(taus)==0:
                    print("Warning: tau list for VFAT {} is empty".format(vfat))
                else:
                    tau_range = (0.5*taus.min(), 2*taus.max())
                    #tau_range = (0, 60e3)
                    tau_ax.flat[iax].hist(taus, bins=20, range=tau_range, alpha=0.5)
                    tau_ax.flat[iax].set_xlabel("Time constant (ns)")
                    tau_ax.flat[iax].set_title("VFAT {}".format(vfat))
                
                    tau_errors_ax.flat[iax].errorbar(
                        efficiency_df["channel"],
                        efficiency_df["tau"]*1e9,
                        yerr=efficiency_df["err_tau"]*1e9, fmt="."
                    )
                    tau_errors_ax.flat[iax].set_ylim(-50, 1000)
                    tau_errors_ax.flat[iax].set_xlabel("VFAT channel")
                    tau_errors_ax.flat[iax].set_ylabel("Time constant (ns)")
                    tau_errors_ax.flat[iax].set_title("VFAT {}".format(vfat))

                    slope_ax.flat[iax].hist(slopes, bins=20, alpha=0.5)
                    slope_ax.flat[iax].set_xlabel("Slope (kHz/µA)")
                    slope_ax.flat[iax].set_title("VFAT {}".format(vfat))

                    tau_slope_ax.flat[iax].errorbar(slopes, taus, yerr=err_taus, fmt=".")
                    tau_slope_ax.flat[iax].set_xlabel("Slope (kHz/µA)")
                    tau_slope_ax.flat[iax].set_ylabel("Time constant (ns)")
                    tau_slope_ax.flat[iax].set_title("VFAT {}".format(vfat))
           
            """cmap_new = mpl.cm.get_cmap("viridis")
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
            )"""

        if not args.strips: rate_ax.flat[iax].legend()
        rate_ax.flat[iax].set_xlabel("X-ray current (µA)")
        rate_ax.flat[iax].set_ylabel("Interacting particle rate (MHz)")
        rate_ax.flat[iax].set_title("VFAT {}".format(vfat))

        efficiency_ax.flat[iax].set_xlabel("Interacting particle rate (kHz/strip)")
        efficiency_ax.flat[iax].set_ylabel("Efficiency")
        efficiency_ax.flat[iax].set_title("VFAT {}".format(vfat))

    nvfats = 4
    nrows = int(nvfats**0.5)
    ncols = int(nvfats/nrows)
    rate_fig, rate_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
    avg_rate_fig, avg_rate_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
    efficiency_fig, efficiency_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
    
    if args.strips:
        tau_fig, tau_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
        tau_errors_fig, tau_errors_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
        slope_fig, slope_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))
        tau_slope_fig, tau_slope_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11*ncols,9*nrows))

    rate_df.groupby("vfat").apply(analyze_rate)
    rate_fig.tight_layout()
    avg_rate_fig.tight_layout()
    efficiency_fig.tight_layout()
    if args.strips: tau_fig.tight_layout()

    rate_df.to_csv(args.odir / "rate.csv", sep=";")
    if args.verbose: print("Rate csv file saved to", args.odir/"rate.csv")
    rate_fig.savefig(args.odir / "rate.png")
    avg_rate_fig.savefig(args.odir / "avg_rate.png")
    if args.verbose: print("Rate plots saved to", args.odir/"rate.png")
    efficiency_fig.savefig(args.odir / "efficiency.png")
    if args.verbose: print("Efficiency plots saved to", args.odir/"efficiency.png")

    if args.strips:
        deadtime_df = pd.DataFrame(deadtime_tuples, columns=["vfat", "channel", "tau", "min_rate"])
        deadtime_df.to_csv(args.odir / "deadtime.csv", sep=";")
        tau_fig.savefig(args.odir / "tau.png")
        tau_errors_fig.savefig(args.odir / "tau_errors.png")
        slope_fig.savefig(args.odir / "slope.png")
        tau_slope_fig.savefig(args.odir / "tau_slope.png")

if __name__=="__main__": main()
