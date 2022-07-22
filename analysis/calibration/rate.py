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
    parser.add_argument("ofile", type=pathlib.Path)
    parser.add_argument("method", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pulse-stretch", type=int, default=7) 
    parser.add_argument("-n", "--events", type=int, default=-1)
    args = parser.parse_args()

    #os.makedirs(args.odir, exist_ok=True)

    if args.method == "daq":
        
        if args.verbose: print("Pulse stretch", args.pulse_stretch)

        with uproot.open(args.ifile) as input_file:
            input_tree = input_file["outputtree"]

            if args.verbose: print("Reading input tree...")
            slots = input_tree["slot"].array(entry_start=args.start, entry_stop=args.events)
            ohs = input_tree["OH"].array(entry_start=args.start, entry_stop=args.events)
            vfats = input_tree["VFAT"].array(entry_start=args.start, entry_stop=args.events)
            channels = input_tree["CH"].array(entry_start=args.start, entry_stop=args.events)
        
            list_slots = np.unique(ak.flatten(slots, axis=None))
            list_oh = np.unique(ak.flatten(ohs, axis=None))
            list_vfat = np.unique(ak.flatten(vfats, axis=None))

            rate_tuples = list()

            total_triggers = ak.num(slots, axis=0)
            if args.verbose: print(f"Taken {total_triggers} triggers ")

            for slot in list_slots:

                ohs_slot = ohs[slots==slot]
                vfats_slot = vfats[slots==slot]
                channels_slot = channels[slots==slot]

                for oh in list_oh:

                    vfats_oh = vfats_slot[ohs_slot==oh]
                    channels_oh = channels_slot[ohs_slot==oh]

                    for vfat in list_vfat:

                        channels_vfat = channels_oh[vfats_oh==vfat]
                        multiplicities = ak.sum(channels_vfat, axis=1)
                        event_count = ak.count_nonzero(multiplicities)
                        event_rate = event_count / (total_triggers * (args.pulse_stretch+1) * 25e-9)
                        event_rate_error = np.sqrt(event_count) / (total_triggers * (args.pulse_stretch+1) * 25e-9)
                        if args.verbose: print("slot {}, oh {}, vfat {}, count {}; rate {}".format(slot, oh, vfat, event_count, event_rate))
                        rate_tuples.append((slot, oh, vfat, event_count, event_rate, event_rate_error, total_triggers))
                        
            rate_df = pd.DataFrame(rate_tuples, columns=["slot", "oh", "vfat", "counts", "rate", "rate_error", "triggers"])
            #rate_df.to_csv(args.odir / "rate.log")

    elif args.method == "sbit":

        rate_df = pd.read_csv(args.ifile, sep=";")
        rate_df["slot"] = 0

    if args.verbose:
        print("Rate dataframe")
        print(rate_df)

    vfat_mapping = {
            ( 0, 1 ): (3, "x"),
            ( 2, 3 ): (3, "y"),
            ( 4, 5 ): (2, "x"),
            ( 6, 7 ): (2, "y"),
            ( 8, 9 ): (1, "x"),
            ( 10, 11 ): (1, "y")
    }

    rate_chamber_tuples = list()

    list_slots = rate_df["slot"].unique()
    for slot in list_slots:
        rate_df_slot = rate_df[rate_df["slot"]==slot]
        if slot == 1:
            total_rate = rate_df_slot["rate"].sum()
            total_rate_error = rate_df_slot["rate_error"].sum()
            if args.verbose: print("ME0 rate", total_rate)
            rate_chamber_tuples.append(("me0", "x", total_rate, total_rate_error))
        elif slot == 0:
            for vfat1, vfat2 in vfat_mapping:

                tracker, direction = vfat_mapping[(vfat1, vfat2)]

                rate_vfat1 = rate_df_slot[rate_df_slot["vfat"]==vfat1]["rate"].iloc[0]
                rate_vfat2 = rate_df_slot[rate_df_slot["vfat"]==vfat2]["rate"].iloc[0]

                rate_vfat1_error = rate_df_slot[rate_df_slot["vfat"]==vfat1]["rate_error"].iloc[0]
                rate_vfat2_error = rate_df_slot[rate_df_slot["vfat"]==vfat2]["rate_error"].iloc[0]

                total_rate = rate_vfat1 + rate_vfat2
                total_rate_error = (rate_vfat1_error**2 + rate_vfat2_error**2)**0.5
                rate_chamber_tuples.append((tracker, direction, total_rate, total_rate_error))

    rate_chamber_df = pd.DataFrame(rate_chamber_tuples, columns=["chamber", "direction", "rate", "rate_error"])
    print(rate_chamber_df)
    rate_chamber_df.to_csv(args.ofile, index=False)

if __name__=='__main__': main()
