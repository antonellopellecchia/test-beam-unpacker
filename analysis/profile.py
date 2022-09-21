import os, sys, pathlib
import argparse
from tqdm import tqdm

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

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", type=pathlib.Path, help="Input track file")
    parser.add_argument('odir', type=pathlib.Path, help="Output directory")
    parser.add_argument("-n", "--events", type=int, default=-1, help="Number of events to analyze")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate logging")
    args = parser.parse_args()
    
    os.makedirs(args.odir, exist_ok=True)

    with uproot.open(args.ifile) as track_file:
        rechit_tree = track_file["rechitTree"]
        if args.verbose: rechit_tree.show()

        print("Reading tree...")
        rechits_chamber = rechit_tree["rechit2DChamber"].array(entry_stop=args.events)
        rechits_x = rechit_tree["rechit2D_X_center"].array(entry_stop=args.events)
        rechits_y = rechit_tree["rechit2D_Y_center"].array(entry_stop=args.events)
        cluster_size_x = rechit_tree["rechit2D_X_clusterSize"].array(entry_stop=args.events)
        cluster_size_y = rechit_tree["rechit2D_Y_clusterSize"].array(entry_stop=args.events)
       
        # choose only events with hits in all chambers:
        #mask_4hit = ak.count_nonzero(rechits_chamber>=0, axis=1)>3
        # keep only events with both cluster sizes below 10:
        mask_cls = (cluster_size_x<=10)&(cluster_size_y<=10)
        rechits_chamber = rechits_chamber[mask_cls]
        rechits_x, rechits_y = rechits_x[mask_cls], rechits_y[mask_cls]
        cluster_size_x, cluster_size_y = cluster_size_x[mask_cls], cluster_size_y[mask_cls]

        chambers_unique = np.unique(ak.flatten(rechits_chamber))
        cls_unique= np.array(np.unique(ak.flatten(cluster_size_x))).astype(int)
        n_chambers = len(chambers_unique)

        if args.verbose:
            n_events = ak.num(rechits_chamber, axis=0)
            print(f"{n_events} events in tree")
            print(f"{n_chambers} chambers in tree")
            print("Chambers:", rechits_chamber)
            print("Rechits x:", rechits_x)
            print("Rechits y:", rechits_y)
                
        # Preparing figures:
        print("Starting plotting...")
        directions = ["x", "y"]
        cls_fig, cls_axs = plt.subplots(nrows=1, ncols=n_chambers, figsize=(11*n_chambers, 9))
        profile_fig, profile_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(11*n_chambers, 9*2))
        profile_cls_fig, profile_cls_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(11*n_chambers, 9*2))
        profile_multiplicity_fig, profile_multiplicity_axs = plt.subplots(nrows=2, ncols=n_chambers, figsize=(11*n_chambers, 9*2))

        for tested_chamber in chambers_unique:
            rechits = [
                rechits_x[rechits_chamber==tested_chamber],
                rechits_y[rechits_chamber==tested_chamber]
            ]
            cluster_sizes = [
                cluster_size_x[rechits_chamber==tested_chamber],
                cluster_size_y[rechits_chamber==tested_chamber]
            ]
            if args.verbose:
                print(f"Processing chamber {tested_chamber}...")
                print("\tRechits x:", rechits[0])
                print("\tRechits y:", rechits[1])
 
            cls_axs[tested_chamber].hist2d(
                ak.flatten(cluster_sizes[0]), ak.flatten(cluster_sizes[1])
                #bins=100, range=(-50,50)
            )
            cls_axs[tested_chamber].set_xlabel("Cluster size x (mm)")
            cls_axs[tested_chamber].set_ylabel("Cluster size y (mm)")
            cls_axs[tested_chamber].set_title(f"Tracker {tested_chamber}")
           
            for idirection in range(2):
                direction = directions[idirection]
                cluster_size = cluster_sizes[idirection]
                cluster_multiplicity = ak.num(rechits[idirection], axis=1)
                mul_unique = np.array(np.unique(cluster_multiplicity)).astype(int)

                if args.verbose:
                    print("\tMultiplicities {} from {} to {}".format(direction, mul_unique.min(), mul_unique.max()))
                    high_multi_filter = cluster_multiplicity>50
                    high_multi_rechits = rechits[idirection][high_multi_filter]
                    print("High multiplicity events:")
                    for mul,r in zip(cluster_multiplicity[high_multi_filter], high_multi_rechits):
                        print("Rechit {} {} mm, multiplicity {}".format(direction, r, mul))

                rechits_unique = np.concatenate([ np.unique(r) for r in rechits[idirection] ])
                print("Rechits unique:", rechits_unique)
                profile_axs[idirection][tested_chamber].hist(
                    #ak.flatten(rechits[idirection]),
                    np.concatenate([ np.unique(r) for r in rechits[idirection] ]),
                    bins=100, range=(-50,50)
                )
                profile_axs[idirection][tested_chamber].set_xlabel(f"Reconstructed {direction} (mm)")
                profile_axs[idirection][tested_chamber].set_title(f"Tracker {tested_chamber}")

                profile_cls_axs[idirection][tested_chamber].hist(
                    [
                        ak.flatten(rechits[idirection][cluster_sizes[idirection]==cls])
                        for cls in cls_unique
                    ],
                    bins=50, range=(-50,50),
                    label=[ f"Cluster size {cls}" for cls in cls_unique ],
                    alpha=0.6, histtype="bar", stacked=True
                )
                profile_cls_axs[idirection][tested_chamber].legend()
                profile_cls_axs[idirection][tested_chamber].set_xlabel(f"Reconstructed {direction} (mm)")
                profile_cls_axs[idirection][tested_chamber].set_title(f"Tracker {tested_chamber}")

                """profile_multiplicity_axs[idirection][tested_chamber].hist(
                    [
                        ak.flatten(rechits[idirection][cluster_multiplicity==mul])
                        for mul in mul_unique
                    ],
                    bins=50, range=(-50,50),
                    label=[ f"Cluster multiplicity {mul}" for mul in mul_unique ],
                    alpha=0.6, histtype="bar", stacked=True
                )"""

                rec, mul = ak.broadcast_arrays(rechits[idirection], cluster_multiplicity)
                profile_multiplicity_axs[idirection][tested_chamber].hist2d(
                    np.array(ak.flatten(rec)), np.array(ak.flatten(mul)),
                    bins=(50,20), range=((-50,50), (0,20))
                )
                #profile_multiplicity_axs[idirection][tested_chamber].legend()
                profile_multiplicity_axs[idirection][tested_chamber].set_xlabel(f"Reconstructed {direction} (mm)")
                profile_multiplicity_axs[idirection][tested_chamber].set_title(f"Tracker {tested_chamber}")

        cls_fig.tight_layout()
        cls_fig.savefig(args.odir / "cluster_size.png")

        profile_fig.tight_layout()
        profile_fig.savefig(args.odir / "profiles.png")

        profile_multiplicity_fig.tight_layout()
        profile_multiplicity_fig.savefig(args.odir / "profiles_multiplicity.png")

        profile_cls_fig.tight_layout()
        profile_cls_fig.savefig(args.odir / "profiles_cluster_size.png")

        print(f"Plots saved to {args.odir}")

if __name__=='__main__': main()
