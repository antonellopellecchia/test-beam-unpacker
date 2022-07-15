import os, sys
import argparse
import pathlib

import numpy as np
import pandas as pd
import awkward as ak
import uproot

from matplotlib import pyplot as plt
from matplotlib import colors
import mplhep as hep
plt.style.use(hep.style.ROOT)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", type=pathlib.Path)
    parser.add_argument("odir", type=pathlib.Path)
    parser.add_argument("-m", "--multiplicities", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--events", type=int, default=-1)
    parser.add_argument("--fed", type=int, default=0)
    parser.add_argument("--slot", type=int)
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)

    with uproot.open(args.ifile) as input_file:
        input_tree = input_file["outputtree"]

        print(input_tree.keys())

        print("Reading input tree...")
        latency = input_tree["runParameter"].array(entry_stop=args.events)
        slots = input_tree["slot"].array(entry_stop=args.events)
        ohs = input_tree["OH"].array(entry_stop=args.events)
        vfats = input_tree["VFAT"].array(entry_stop=args.events)
        channels = input_tree["CH"].array(entry_stop=args.events)
        strips = input_tree["digiStrip"].array(entry_stop=args.events)

        slot_mask = slots==args.slot
        slots, ohs, vfats, strips = slots[slot_mask], ohs[slot_mask], vfats[slot_mask], strips[slot_mask]

        print("Slots:", np.unique(ak.flatten(slots)))
        print("OH:", np.unique(ak.flatten(ohs)))
        print("Latency:", latency)

        if args.multiplicities:
            
            for oh in np.unique(ak.flatten(ohs)):
                        
                vfats_oh = vfats[ohs==oh]
                channels_oh = channels[ohs==oh]
                
                latency_multiplicity_fig = plt.figure(figsize=(12*4,10*6))
                for vfat in np.unique(ak.flatten(vfats_oh)):

                    channels_vfat = channels_oh[vfats_oh==vfat]
                    print(f"channels vfat {vfat}", channels_vfat)
                    multiplicities = ak.count(channels_vfat, axis=1)
                    count = ak.count_nonzero(multiplicities)

                    print("latency", latency, len(latency))
                    print("latency", multiplicities, len(multiplicities))

                    latency_multiplicities_ax = latency_multiplicity_fig.add_subplot(6, 4, vfat+1)
                    latency_multiplicities_ax.hist(multiplicities[latency<=52], label="latency < 52", alpha=0.3, bins=128, range=(0,128))
                    latency_multiplicities_ax.hist(multiplicities[latency>52], label="latency > 52", alpha=0.3, bins=128, range=(0,128))
                    latency_multiplicities_ax.legend()
                    latency_multiplicities_ax.set_xlabel("Event multiplicity")
                    latency_multiplicities_ax.set_yscale("log")
                    #latency_multiplicities_ax.hist2d(np.array(latency), np.array(multiplicities), range=((45,60),(0,100)), norm=colors.LogNorm())
                    #latency_multiplicities_ax.set_xlabel("Latency (BX)")
                    #latency_multiplicities_ax.set_ylabel("Event multiplicity")
                    #latency_multiplicities_ax.set_zlim(1, 10e3)
                
                latency_multiplicity_fig.tight_layout()
                latency_multiplicity_fig.savefig(args.odir / "latency_multiplicity_fed{}_oh{}.png".format(args.fed, oh))

        latency_tuples = list()
        for fixed_latency in np.unique(latency):
            oh_latency = ohs[latency==fixed_latency]
            vfat_latency = vfats[latency==fixed_latency]
            strips_latency = strips[latency==fixed_latency]
            
            for oh in np.unique(ak.flatten(ohs)):
                
                vfats_oh = vfat_latency[oh_latency==oh]
                strips_oh = strips_latency[oh_latency==oh]
                for vfat in np.unique(ak.flatten(vfats_oh)):

                    strips_vfat = strips_oh[vfats_oh==vfat]
                    multiplicities = ak.sum(strips_vfat, axis=1)
                    count = ak.count_nonzero(multiplicities)

                    if args.verbose: print("oh {}, vfat {}, latency {}; count {}".format(oh, vfat, fixed_latency, count))
                    latency_tuples.append((oh, vfat, fixed_latency, count))
                    
        latency_df = pd.DataFrame(latency_tuples, columns=["oh", "vfat", "latency", "counts"])
        latency_df.to_csv(args.odir / "latencies.log")
        print(latency_df)


        latency_fig, latency_ax = plt.figure(figsize=(12,10)), plt.axes() 
        latency_ax.bar(latency_df["latency"], latency_df["counts"], width=2, alpha=0.4)
        latency_ax.set_xlim(30, 80)
        latency_ax.set_xlabel("Latency (BX)")
        latency_ax.set_ylabel("Counts")
        latency_fig.savefig(args.odir / "latency.png")
        
        latency_figs = {
            oh : plt.figure(figsize=(4*12, 6*10)) 
            for oh in latency_df["oh"].unique()
        }
  
        def find_optimal_latency(df):
        
            df = df.reset_index()
            oh, vfat = df["oh"].iloc[0], df["vfat"].iloc[0]
            latencies, counts = df["latency"], df["counts"]
            #optimal_latency = int(latencies[np.argmax(counts)])
            optimal_latency = (np.sum(latencies*counts)/np.sum(counts)).astype(int)
            
            latency_ax = latency_figs[oh].add_subplot(6, 4, vfat+1)
            print("oh {}, vfat {}, optimal latency {}".format(oh, vfat, optimal_latency))
            latency_ax.bar(latencies, counts, width=1, alpha=0.4, label=f"{optimal_latency} BX".format(optimal_latency))
            latency_ax.set_xlim(30, 80)
            latency_ax.set_xlabel("Latency (BX)")
            latency_ax.set_ylabel("Counts")
            latency_ax.set_title("OH {}, VFAT {}".format(oh, vfat))
            latency_ax.legend()
            return optimal_latency

        latency_groups = latency_df.groupby(["oh", "vfat"])
        optimal_latencies = latency_groups.apply(find_optimal_latency)
        
        for oh in latency_figs:
            latency_figs[oh].savefig(args.odir / "latency_fed{}_oh{}.png".format(args.fed, oh))

        optimal_latencies.rename("latency", inplace=True)
        latencies_df = optimal_latencies.to_frame()
        latencies_df["fed"] = args.fed
        latencies_df["slot"] = 0 # hack, to be improved
        print(latencies_df)
        latencies_df.to_csv(args.odir / "latencies.cfg", sep=";")

if __name__=='__main__': main()
