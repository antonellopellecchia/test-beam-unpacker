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

def f_efficiency(x, zero_eff, tau):
    return zero_eff / (1 + x*tau)

def efficiency_fit(rate, efficiency, err_efficiency):
    p0 = [max(efficiency), 0]
    popt, pcov = curve_fit(f_efficiency, rate, efficiency, sigma=err_efficiency, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scandir", type=pathlib.Path, help="Scan input directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log level")
    args = parser.parse_args()
    
    scan_df = pd.read_csv(args.scandir / "runs.csv")
    if args.verbose: print("Runs:\n", scan_df)

    scan_df_nocomp = scan_df[scan_df["compensation"]=="no"]
    scan_df_comp = scan_df[scan_df["compensation"]=="yes"]
    
    efficiency_fig, efficiency_ax = plt.figure(figsize=(12,9)), plt.axes()
    rate_nocomp, rate_comp = 1/scan_df_nocomp["source"], 1/scan_df_comp["source"]
    efficiency_ax.errorbar(rate_nocomp, scan_df_nocomp["efficiency"], scan_df_nocomp["err_efficiency"], fmt="o", label="No compensation", color="red")
    efficiency_ax.errorbar(rate_comp, scan_df_comp["efficiency"], scan_df_comp["err_efficiency"], fmt="o", label="Compensation", color="blue")

    rate_x = np.linspace(min(rate_nocomp), max(rate_nocomp), 100) 
    nocomp_pars, nocomp_errs = efficiency_fit(rate_nocomp, scan_df_nocomp["efficiency"], scan_df_nocomp["err_efficiency"])
    efficiency_ax.plot(rate_x, f_efficiency(rate_x, *nocomp_pars), "--", color="red")
    comp_pars, comp_errs = efficiency_fit(rate_comp, scan_df_comp["efficiency"], scan_df_comp["err_efficiency"])
    efficiency_ax.plot(rate_x, f_efficiency(rate_x, *comp_pars), "--", color="blue")

    tau_nocomp, err_tau_nocomp = nocomp_pars[1]/200e3*1e9, nocomp_errs[1]/200e3*1e9
    tau_comp, err_tau_comp = comp_pars[1]/200e3*1e9, comp_errs[1]/200e3*1e9
    print("No compensation tau {0:1.1f} ± {1:1.1f} ns".format(tau_nocomp, err_tau_nocomp))
    print("Compensation tau {0:1.1f} ± {1:1.1f} ns".format(tau_comp, err_tau_comp))

    efficiency_ax.set_xlabel("Inverse source absorption factor")
    efficiency_ax.set_ylabel("Efficiency")
    efficiency_ax.set_xscale("log")
    efficiency_ax.set_ylim(0.85, 1.)
    efficiency_ax.legend()
    efficiency_fig.savefig(args.scandir / "rate_capability.png")
    efficiency_fig.savefig(args.scandir / "rate_capability.pdf")

    resolution_fig, resolution_ax = plt.figure(figsize=(12,9)), plt.axes()
    resolution_ax.errorbar(1/scan_df_nocomp["source"], scan_df_nocomp["space_resolution"], scan_df_nocomp["err_space_resolution"], fmt="o--", label="No compensation", color="red")
    resolution_ax.errorbar(1/scan_df_comp["source"], scan_df_comp["space_resolution"], scan_df_comp["err_space_resolution"], fmt="o--", label="Compensation", color="blue")
    resolution_ax.set_xlabel("Inverse source absorption factor")
    resolution_ax.set_ylabel("Space resolution (µm)")
    resolution_ax.set_xscale("log")
    resolution_ax.legend()
    resolution_fig.savefig(args.scandir / "space_resolution.png")
    resolution_fig.savefig(args.scandir / "space_resolution.pdf")

    cls_fig, cls_ax = plt.figure(figsize=(12,9)), plt.axes()
    cls_ax.errorbar(1/scan_df_nocomp["source"], scan_df_nocomp["cls_muon"], scan_df_nocomp["err_cls_muon"], fmt="o--", label="Muons no compensation", color="red")
    cls_ax.errorbar(1/scan_df_comp["source"], scan_df_comp["cls_muon"], scan_df_comp["err_cls_muon"], fmt="o--", label="Muons compensation", color="blue")
    cls_ax.errorbar(1/scan_df_nocomp["source"], scan_df_nocomp["cls_bkg"], scan_df_nocomp["err_cls_bkg"], fmt="o--", label="Background no compensation", color="purple")
    cls_ax.errorbar(1/scan_df_comp["source"], scan_df_comp["cls_bkg"], scan_df_comp["err_cls_bkg"], fmt="o--", label="Background compensation", color="green")
    cls_ax.set_xlabel("Inverse source absorption factor")
    cls_ax.set_ylabel("Cluster size")
    cls_ax.set_xscale("log")
    cls_ax.legend()
    cls_fig.savefig(args.scandir / "cluster_size.png")
    cls_fig.savefig(args.scandir / "cluster_size.pdf")

if __name__=='__main__': main()
