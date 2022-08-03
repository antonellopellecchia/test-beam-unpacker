import os, sys
import argparse
import pathlib

import numpy as np
import pandas as pd
import awkward as ak
import uproot

from matplotlib import pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", type=pathlib.Path)
    parser.add_argument("odir", type=pathlib.Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("-n", "--events", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)

    with uproot.open(args.ifile) as input_file:
        input_tree = input_file["outputtree"]

        print(input_tree.keys())

        print("Reading input tree...")
        thresholds = input_tree["runParameter"].array(entry_start=args.start, entry_stop=args.events)
        ohs = input_tree["OH"].array(entry_start=args.start, entry_stop=args.events)
        vfats = input_tree["VFAT"].array(entry_start=args.start, entry_stop=args.events)
        strips = input_tree["digiStrip"].array(entry_start=args.start, entry_stop=args.events)

        print(thresholds)
    
        #print("Calculating multiplicities..")
        #event_count = ak.count(multiplicities>0)
       
        #filter_threshold = (thresholds%5==1)&(thresholds<110)
        filter_threshold = (thresholds>0)&(thresholds<110)
       
        #if not fixed_threshold % 5 == 0: continue
        #if fixed_threshold > 110: continue
        thresholds, ohs, vfats, strips = thresholds[filter_threshold], ohs[filter_threshold], vfats[filter_threshold], strips[filter_threshold]

        list_oh = np.unique(ak.flatten(ohs, axis=None))
        list_vfat = np.unique(ak.flatten(vfats, axis=None))

        threshold_tuples = list()
        scan_thresholds = np.unique(thresholds)
        print("Analyzing threshold range", scan_thresholds)
        for fixed_threshold in scan_thresholds:

            oh_threshold = ohs[thresholds==fixed_threshold]
            vfat_threshold = vfats[thresholds==fixed_threshold]
            strips_threshold = strips[thresholds==fixed_threshold]
            
            total_triggers = ak.num(oh_threshold, axis=0)
            print("{} triggers for threshold {}".format(total_triggers, fixed_threshold))

            for oh in list_oh:
               
                print(vfat_threshold)
                vfats_oh = vfat_threshold[oh_threshold==oh]
                strips_oh = strips_threshold[oh_threshold==oh]
                
                for vfat in list_vfat:

                    #selection = (thresholds==fixed_threshold)&(ohs==oh)&(vfats==vfat)
                    strips_selected = strips_oh[vfats_oh==vfat]
                    multiplicities = ak.sum(strips_selected, axis=1)
                    event_count = ak.count_nonzero(multiplicities)
                    event_rate = event_count / (total_triggers * 200e-9)
                    print(f"Analyzing threshold {fixed_threshold}, oh {oh}, vfat {vfat}")
                    #print("strips", strips)
                    #print("multiplicity", multiplicities)
                    #print("count", event_count)
                    if args.verbose: print("oh {}, vfat {}, threshold {}; count {}; rate {}".format(oh, vfat, fixed_threshold, event_count, event_rate))
                    threshold_tuples.append((oh, vfat, fixed_threshold, event_count, event_rate))
                    
        threshold_df = pd.DataFrame(threshold_tuples, columns=["oh", "vfat", "threshold", "counts", "rate"])
        threshold_df.to_csv(args.odir / "thresholds.log")
        print("OptoHybrid list:", threshold_df["oh"].unique())

        for oh in threshold_df["oh"].unique():
            
            threshold_df_oh = threshold_df[threshold_df["oh"]==oh]

            print(f"Plotting threshold scan for OH {oh}")

            def plot_thresholds_vfat(df):

                vfat = df["vfat"].iloc[0]
                thresholds, counts, rate = df["threshold"], df["counts"], df["rate"]

                try:
                    threshold_ax = threshold_fig.add_subplot(3, 4, vfats_map[vfat]+1)
                except ValueError:
                    print(f"No data for VFAT {vfat}, skipping...")
                    return None
                threshold_ax.plot(thresholds, rate, "o-")
                threshold_ax.set_title(f"OH {oh} VFAT {vfat}")
                threshold_ax.set_xlabel("THR_ARM_DAC")
                threshold_ax.set_ylabel("Rate (Hz)")
               
                print(thresholds[rate<1e2])
                optimal_threshold = thresholds[rate<1e2].min()
                print(f"Found optimal threshold {optimal_threshold} for vfat {vfat}")
                return optimal_threshold

            vfats_unique = threshold_df_oh["vfat"].unique()
            vfats_map = { vfat:i for i, vfat in enumerate(vfats_unique) }

            threshold_fig = plt.figure(figsize=(12*4, 10*3))
            print(threshold_df_oh)
            threshold_groups = threshold_df_oh.groupby(["vfat"])
            threshold_df = pd.DataFrame(threshold_groups.apply(plot_thresholds_vfat))

            threshold_df["threshold"] = threshold_df[0]
            threshold_df["vfat"] = threshold_df.index
            threshold_df["fed"] = 10
            threshold_df["slot"] = 0
            threshold_df["oh"] = oh
            threshold_df = threshold_df[["fed", "slot", "oh", "vfat", "threshold"]]
            print(threshold_df)
            threshold_df.to_csv(args.odir / f"threshold_scan_oh{oh}.csv", sep=";", index=False)
            
            threshold_fig.tight_layout()
            threshold_fig.savefig(args.odir / f"threshold_scan_oh{oh}.png")

if __name__=='__main__': main()
