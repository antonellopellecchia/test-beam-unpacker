"""
Plot transversal or angular corrections vs iteration
"""

import os, sys, pathlib
import argparse, re

from tqdm import tqdm

import uproot
import numpy as np
import awkward as ak
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

correction_parameters = [
    ("translation_x", "err_translation_x", "x correction (mm)", 1),
    ("translation_y", "err_translation_y", "y correction (mm)", 1),
    ("angle", "err_angle", "Angle correction (mrad)", 1e3),
]

markers = [ "o", "s", "h", "v" ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idirs", type=pathlib.Path, nargs="+", help="Input directory")
    parser.add_argument("odir", type=pathlib.Path, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.odir, exist_ok=True)

    corrections = list()
    for idir in args.idirs:
        match = re.search("iteration_(\d+)/chamber_(\d+)", str(idir))
        iteration, chamber = int(match.group(1)), int(match.group(2))
        if args.verbose: print("Reading iteration {}, chamber {}...".format(iteration, chamber))
        correction_step_df = pd.read_csv(os.path.join(idir, "corrections.txt"), sep=";", index_col=0)
        correction_step_df["chamber"] = correction_step_df.index
        correction_step_df.reset_index(level=0, inplace=True, drop=True)
        correction_step_df["iteration"] = iteration
        correction_step = correction_step_df[correction_step_df["chamber"]==chamber]
        corrections.append(correction_step)

    corrections_df = pd.concat(corrections)
    corrections_df.reset_index()
    print("Correction dataframe:")
    print(corrections_df)

    corrections_df.to_csv(args.odir / "corrections.csv", index=False, sep=";")

    for corr_name, corr_errname, corr_label, corr_scale in correction_parameters:

        def plot_corrections(df):

            chamber = df.iloc[0]["chamber"].astype(int)
            iterations = df["iteration"]
            corrections, err_corrections = df[corr_name], df[corr_errname]
            corrections = corrections - corrections.iloc[-1]
            ax.plot(iterations, corrections*corr_scale, "-"+markers[chamber], label=f"Tracker {chamber+1}", markersize=10)
            ax.fill_between(iterations, corr_scale*(corrections-err_corrections), corr_scale*(corrections+err_corrections), alpha=0.3)
            ax.set_xticks(iterations)
            #signed_corrections = [ (-1)**n*correction for n,correction in enumerate(corrections) ]
            #total_correction = np.sum(signed_corrections)
            #print("Chamber {}, {} {}".format(chamber, corr_name, total_correction))

        fig, ax = plt.figure(figsize=(12, 10)), plt.axes()

        corrections_df.groupby("chamber").apply(plot_corrections)

        ax.set_xlabel("Iteration")
        ax.set_ylabel(corr_label)
        ax.legend()
        fig.savefig(args.odir / f"{corr_name}.png")

if __name__=="__main__": main()
