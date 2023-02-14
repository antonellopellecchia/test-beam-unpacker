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

def clusters_to_digis(cluster_first, cluster_size):
    events = list()
    for event_first,event_size in zip(cluster_first, cluster_size):
        clusters = list()
        for first,size in zip(event_first, event_size):
            cluster = np.ones(size)*first
            cluster += np.linspace(0, size-1, size)
            clusters.append(cluster)
        events.append(clusters)
    return ak.Array(events)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", type=pathlib.Path, help="Input track file")
    parser.add_argument('odir', type=pathlib.Path, help="Output directory")
    parser.add_argument("-n", "--events", type=int, default=-1, help="Number of events to analyze")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate logging")
    parser.add_argument("--debug", action="store_true", help="Debug mapping")
    args = parser.parse_args()
    
    os.makedirs(args.odir, exist_ok=True)

    with uproot.open(args.ifile) as track_file:
        rechit_tree = track_file["rechitTree"]
        if args.verbose: rechit_tree.show()

        print("Reading tree...")
        cluster_chamber = rechit_tree["clusterChamber"].array(entry_stop=args.events)
        cluster_eta = rechit_tree["clusterEta"].array(entry_stop=args.events)
        cluster_center = rechit_tree["clusterCenter"].array(entry_stop=args.events)
        cluster_first = rechit_tree["clusterFirst"].array(entry_stop=args.events)
        cluster_size = rechit_tree["clusterSize"].array(entry_stop=args.events)

        # choose only events with hits in all chambers:
        #mask_4hit = ak.count_nonzero(cluster_chamber>=0, axis=1)>3
        # keep only events with both cluster sizes below 10:
        mask_cls = (cluster_size<=10)
        cluster_chamber = cluster_chamber[mask_cls]
        cluster_center= cluster_center[mask_cls]
        cluster_size = cluster_size[mask_cls]

        chambers_unique = np.unique(ak.flatten(cluster_chamber))
        cls_unique = np.array(np.unique(ak.flatten(cluster_size))).astype(int)
        n_chambers = len(chambers_unique)

        if args.verbose:
            n_events = ak.num(cluster_chamber, axis=0)
            print(f"{n_events} events in tree")
            print(f"{n_chambers} chambers in tree")
            print("Chambers:", cluster_chamber)
            print("Cluster center:")
                
        # Preparing figures:
        print("Starting plotting...")
        etas = np.unique(ak.flatten(cluster_eta).to_numpy())
        n_etas = len(etas)
        cls_fig, cls_axs = plt.subplots(nrows=n_etas, ncols=n_chambers, figsize=(11*n_chambers, 9*n_etas))
        profile_fig, profile_axs = plt.subplots(nrows=n_etas, ncols=n_chambers, figsize=(11*n_chambers, 9*n_etas))
        profile_cls_fig, profile_cls_axs = plt.subplots(nrows=n_etas, ncols=n_chambers, figsize=(11*n_chambers, 9*n_etas))
        profile_cls_stacked_fig, profile_cls_stacked_axs = plt.subplots(nrows=n_etas, ncols=n_chambers, figsize=(11*n_chambers, 9*n_etas))
        profile_multiplicity_fig, profile_multiplicity_axs = plt.subplots(nrows=n_etas, ncols=n_chambers, figsize=(11*n_chambers, 9*n_etas))
        profile_2events_fig, profile_2events_axs = plt.subplots(nrows=n_etas, ncols=n_chambers, figsize=(11*n_chambers, 9*n_etas))
        clusters_event_fig, clusters_event_axs = plt.subplots(nrows=n_etas, ncols=n_chambers, figsize=(11*n_chambers, 9*n_etas))

        for tested_chamber in chambers_unique:
            filter_chamber = cluster_chamber == tested_chamber
            cluster_center_chamber = cluster_center[filter_chamber]
            cluster_eta_chamber = cluster_eta[filter_chamber]
            cluster_first_chamber = cluster_first[filter_chamber]
            cluster_size_chamber = cluster_size[filter_chamber]
            etas_chamber = np.unique(ak.flatten(cluster_eta_chamber))
            
            if args.verbose:
                print(f"Processing chamber {tested_chamber}...")
                print("\tCluster center:", clusters_center)
 
            for ieta, eta in enumerate(etas_chamber):

                filter_eta = cluster_eta_chamber == eta
                cluster_center_eta = cluster_center_chamber[filter_eta]
                cluster_size_eta = cluster_size_chamber[filter_eta]
                cluster_first_eta = cluster_first_chamber[filter_eta]
                cluster_multiplicity = ak.num(cluster_center_eta, axis=1)
                mul_unique = np.array(np.unique(cluster_multiplicity)).astype(int)

                cluster_bins = ak.max(cluster_center_eta)-ak.min(cluster_center_eta)+1
                cluster_range = (ak.min(cluster_center_eta)-0.5, ak.max(cluster_center_eta)+0.5)

                cls_axs[ieta][tested_chamber].hist(
                    ak.flatten(cluster_size_eta),
                    histtype="step", linewidth=2, color="red"
                )
                cls_axs[ieta][tested_chamber].set_xlabel("Cluster size (mm)")
                cls_axs[ieta][tested_chamber].set_title(f"Detector {tested_chamber} eta {eta}")
                
                if args.verbose:
                    print("\tMultiplicities {} from {} to {}".format(direction, mul_unique.min(), mul_unique.max()))
                    high_multi_filter = cluster_multiplicity>50
                    high_multi_rechits = cluster_center_eta[high_multi_filter]
                    print("High multiplicity events:")
                    for mul,r in zip(cluster_multiplicity[high_multi_filter], high_multi_rechits):
                        print("Cluster eta {} {} mm, multiplicity {}".format(eta, r, mul))
                    clusters_unique = np.concatenate([ np.unique(r) for r in cluster_center_eta ])
                    print("Rechits unique:", clusters_unique)

                profile_axs[ieta][tested_chamber].hist(
                    ak.flatten(cluster_center_eta),
                    #np.concatenate([ np.unique(r) for r in rechits[ieta] ]),
                    bins=cluster_bins, range=cluster_range
                )
                profile_axs[ieta][tested_chamber].set_xlabel(f"Cluster center")
                profile_axs[ieta][tested_chamber].set_title(f"Detector {tested_chamber} eta {eta}")

                profile_cls_axs[ieta][tested_chamber].hist(
                    [
                        ak.flatten(cluster_center_eta[cluster_size_eta==cls])
                        for cls in cls_unique
                    ],
                    bins=cluster_bins, range=cluster_range,
                    label=[ f"Cluster size {cls}" for cls in cls_unique ],
                    histtype="step", linewidth=2
                )
                profile_cls_axs[ieta][tested_chamber].legend()
                profile_cls_axs[ieta][tested_chamber].set_xlabel(f"Cluster center")
                profile_cls_axs[ieta][tested_chamber].set_title(f"Detector {tested_chamber} eta {eta}")

                profile_cls_stacked_axs[ieta][tested_chamber].hist(
                    [
                        ak.flatten(cluster_center_eta[cluster_size_eta==cls])
                        for cls in cls_unique
                    ],
                    bins=cluster_bins, range=cluster_range,
                    label=[ f"Cluster size {cls}" for cls in cls_unique ],
                    alpha=0.6, histtype="bar", stacked=True
                )
                profile_cls_stacked_axs[ieta][tested_chamber].legend()
                profile_cls_stacked_axs[ieta][tested_chamber].set_xlabel(f"Cluster center")
                profile_cls_stacked_axs[ieta][tested_chamber].set_title(f"Detector {tested_chamber} eta {eta}")
    
                """profile_multiplicity_axs[ieta][tested_chamber].hist(
                    [
                        ak.flatten(rechits[ieta][cluster_multiplicity==mul])
                        for mul in mul_unique
                    ],
                    bins=50, range=(-50,50),
                    label=[ f"Cluster multiplicity {mul}" for mul in mul_unique ],
                    alpha=0.6, histtype="bar", stacked=True
                )"""

                clus, mul = ak.broadcast_arrays(cluster_center_eta, cluster_multiplicity)
                profile_multiplicity_axs[ieta][tested_chamber].hist2d(
                    np.array(ak.flatten(clus)), np.array(ak.flatten(mul)),
                    bins=(cluster_bins,20), range=(cluster_range, (0,20))
                )
                profile_multiplicity_axs[ieta][tested_chamber].set_xlabel(f"Cluster center")
                profile_multiplicity_axs[ieta][tested_chamber].set_ylabel(f"Event multiplicity")
                profile_multiplicity_axs[ieta][tested_chamber].set_title(f"Detector {tested_chamber} eta {eta}")

                """ Plot events with multiplicity 2,
                on x the first hit, on y the second.
                Useful to debug the mapping """
                filter_multi2 = cluster_multiplicity==2
                centers_multi2 = cluster_center_eta[filter_multi2]
                first_multi2 = cluster_first_eta[filter_multi2]
                last_multi2 = first_multi2 + cluster_size_eta[filter_multi2]
                print("Chamber {} eta {}, events with multi 2: {}".format(tested_chamber, eta, centers_multi2))
                centers_multi2_pairs = centers_multi2.to_numpy().transpose()
                first_multi2_pairs = first_multi2.to_numpy().transpose()
                last_multi2_pairs = last_multi2.to_numpy().transpose()
                h, xedges, yedges, _ = profile_2events_axs[ieta][tested_chamber].hist2d(
                    centers_multi2_pairs[0], last_multi2_pairs[1],
                    bins=(cluster_bins, cluster_bins), range=(cluster_range, cluster_range)
                )
                max_index = h.argmax()
                max_x, max_y = int(max_index/h.shape[0]), max_index%h.shape[1]
                print(max_x, max_y, h.max(), h[max_x][max_y])
                profile_2events_axs[ieta][tested_chamber].plot(
                    np.linspace(cluster_range[0], cluster_range[-1]),
                    np.linspace(cluster_range[0], cluster_range[-1]),
                    "-", color="red"
                )
                profile_2events_axs[ieta][tested_chamber].set_xlabel(f"Center first cluster ")
                profile_2events_axs[ieta][tested_chamber].set_ylabel(f"Center second cluster")
                profile_2events_axs[ieta][tested_chamber].set_title(f"Detector {tested_chamber} eta {eta}")

                if args.debug:
                    """ Plot cluster centers vs events 
                    for the first n_events events
                    Useful to debug wrong clusters """
                    n_events = 100
                    multiplicity_filter = ak.num(cluster_center_eta, axis=1)==2 # change this for higher cluster multiplicities
                    digis = clusters_to_digis(cluster_first_eta, cluster_size_eta)
                    #clusters_limited = cluster_center_eta[multiplicity_filter][:n_events]
                    digis = digis[multiplicity_filter][:n_events]
                    for ev in digis: print(ev)
                    event_range = np.arange(len(digis))
                    event_broad, _ = ak.broadcast_arrays(event_range, digis)

                    clusters_event_axs[ieta][tested_chamber].hist2d(
                        ak.flatten(digis, axis=None).to_numpy(),
                        ak.flatten(event_broad, axis=None).to_numpy(),
                        bins=(cluster_bins,n_events),
                        range=[cluster_range,(-0.5,n_events-0.5)]
                    )
                    clusters_event_axs[ieta][tested_chamber].set_xlabel("Cluster center")
                    clusters_event_axs[ieta][tested_chamber].set_ylabel("Event number")
                    clusters_event_axs[ieta][tested_chamber].set_title(f"Detector {tested_chamber} eta {eta}")

        cls_fig.tight_layout()
        cls_fig.savefig(args.odir / "cluster_size.png")

        profile_fig.tight_layout()
        profile_fig.savefig(args.odir / "profiles.png")

        profile_multiplicity_fig.tight_layout()
        profile_multiplicity_fig.savefig(args.odir / "profiles_multiplicity.png")

        profile_2events_fig.tight_layout()
        profile_2events_fig.savefig(args.odir / "profiles_2events.png")

        profile_cls_fig.tight_layout()
        profile_cls_fig.savefig(args.odir / "profiles_cluster_size.png")

        profile_cls_stacked_fig.tight_layout()
        profile_cls_stacked_fig.savefig(args.odir / "profiles_cluster_size_stacked.png")

        clusters_event_fig.tight_layout()
        clusters_event_fig.savefig(args.odir / "clusters_by_events.png")

        print(f"Plots saved to {args.odir}")

if __name__=='__main__': main()
