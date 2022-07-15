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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", type=pathlib.Path, help="Input file")
    parser.add_argument('odir', type=pathlib.Path, help="Output directory")
    parser.add_argument("-n", "--events", type=int, default=-1, help="Number of events to analyse")
    parser.add_argument("--find-noisy", type=pathlib.Path, help="Find noisy strips and save them to file")
    parser.add_argument("--mask-noisy", type=pathlib.Path, help="Mask noisy strips found in file")
    parser.add_argument("--efficiency", type=pathlib.Path, nargs="+", help="Save fast efficiency to file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate logging")
    args = parser.parse_args()
    
    os.makedirs(args.odir, exist_ok=True)

    with uproot.open(args.ifile) as digi_file:
        digi_tree = digi_file["outputtree"]
        if args.verbose: digi_tree.show()

        digi_slot = digi_tree["slot"].array(entry_stop=args.events)
        digi_oh = digi_tree["OH"].array(entry_stop=args.events)
        digi_vfat = digi_tree["VFAT"].array(entry_stop=args.events)
        digi_channel = digi_tree["CH"].array(entry_stop=args.events)
        digi_chamber = digi_tree["digiChamber"].array(entry_stop=args.events)
        digi_eta = digi_tree["digiEta"].array(entry_stop=args.events)
        digi_strip = digi_tree["digiStrip"].array(entry_stop=args.events)

        n_events = ak.num(digi_slot, axis=0)
        print(n_events, "events in run")

        if args.find_noisy: # save to csv strips with occupancy > 4000, to be improved
            os.makedirs(os.path.dirname(args.find_noisy), exist_ok=True)
            noisy_channels = list()

        if args.mask_noisy:
            noisy_df = pd.read_csv(args.mask_noisy, sep=";")
            noisy_mask = False
            for irow, noisy_row in noisy_df.iterrows():
                slot, oh, vfat, channel = noisy_row[["slot", "oh", "vfat", "channel"]]
                if args.verbose: print("Masking slot {}, oh {}, vfat {}, channel {}".format(slot, oh, vfat, channel))
                noisy_mask = noisy_mask|((digi_slot==slot)&(digi_oh==oh)&(digi_vfat==vfat)&(digi_channel==channel))
            noisy_mask = ~noisy_mask
            digi_slot, digi_oh, digi_vfat, digi_channel = digi_slot[noisy_mask], digi_oh[noisy_mask], digi_vfat[noisy_mask], digi_channel[noisy_mask]
            digi_chamber, digi_eta, digi_strip = digi_chamber[noisy_mask], digi_eta[noisy_mask], digi_strip[noisy_mask]

        if args.efficiency:
            for d in args.efficiency:
                os.makedirs(os.path.dirname(d), exist_ok=True)
            event_count_oh = list()
            event_count_chamber = list()

        """ Plot occupancy per strip, i.e. after mapping """
        for slot in np.unique(ak.flatten(digi_slot)):

            slot_filter = digi_slot==slot
            for oh in np.unique(ak.flatten(digi_oh)):
                
                occupancy_fig, occupancy_axs = plt.subplots(nrows=6, ncols=4, figsize=(12*4,10*6))
                occupancy_axs = occupancy_axs.flatten()

                oh_filter = slot_filter&(digi_oh==oh)

                if args.efficiency:
                    oh_channels = digi_channel[oh_filter]
                    channels_per_event = ak.sum(oh_channels, axis=1)
                    good_events = ak.count_nonzero(channels_per_event)
                    event_count_oh.append( (slot, oh, good_events) )

                for vfat in np.unique(ak.flatten(digi_vfat)):

                    vfat_filter = oh_filter&(digi_vfat==vfat)
                    filtered_channel = ak.flatten(digi_channel[vfat_filter])
                    filtered_oh = ak.flatten(digi_oh[vfat_filter])
                    filtered_vfat = ak.flatten(digi_vfat[vfat_filter])

                    if args.verbose: print(f"slot {slot}, oh {oh}, vfat {vfat}, channels:", filtered_channel)
                    
                    if ak.count(filtered_channel) == 0: continue # no hits for selected oh, vfat
                   
                    channel_hist, channel_bins, _ = occupancy_axs[vfat].hist(filtered_channel, bins=140, range=(-0.5,139.5))
                    occupancy_axs[vfat].set_title("OH {}, VFAT {}".format(oh, vfat)) 
                    occupancy_axs[vfat].set_xlabel("Channel")
                    
                    if args.find_noisy:
                        channels = (0.5*(channel_bins[1:]+channel_bins[:-1])).astype(int)
                        noisy_channels_vfat = channels[channel_hist>4500]
                        for channel in noisy_channels_vfat:
                            noisy_channels.append( (slot, oh, vfat, channel) )
               
                occupancy_fig.tight_layout()
                occupancy_fig.savefig(args.odir / f"occupancy_slot{slot}_oh{oh}.png")

        if args.find_noisy:
            noisy_df = pd.DataFrame(noisy_channels, columns=["slot", "oh", "vfat", "channel"])
            noisy_df.to_csv(args.find_noisy, sep=";", index=False)

        if args.efficiency:
            efficiency_df = pd.DataFrame(event_count_oh, columns=["slot", "oh", "fired"])
            efficiency_df["efficiency"] = efficiency_df["fired"]/n_events
            efficiency_df.to_csv(args.efficiency[0], sep=";", index=False)

        """ Plot occupancy per strip, i.e. after mapping """
        for chamber in np.unique(ak.flatten(digi_chamber)):
            
            chamber_filter = digi_chamber==chamber

            if args.efficiency:
                chamber_strips = digi_strip[chamber_filter]
                strips_per_event = ak.sum(chamber_strips, axis=1)
                good_events = ak.count_nonzero(strips_per_event)
                event_count_chamber.append( (chamber, good_events) )

            for eta in np.unique(ak.flatten(digi_eta)):

                eta_filter = digi_eta==eta
                filtered_strips = ak.flatten(digi_strip[(chamber_filter)&(eta_filter)])
                filtered_oh = ak.flatten(digi_oh[(chamber_filter)&(eta_filter)])
                filtered_vfat = ak.flatten(digi_vfat[(chamber_filter)&(eta_filter)])

                if args.verbose: print(f"chamber {chamber}, eta {eta}, strips:", filtered_strips)
                
                if ak.count(filtered_strips) == 0: continue # no vfats for selected chamber, eta
               
                oh = np.unique(filtered_oh)[0]
                vfats = np.unique(filtered_vfat)

                occupancy_fig, occupancy_ax = plt.figure(figsize=(12,10)), plt.axes()
                for vfat in vfats:
                    filtered_strips_vfat = filtered_strips[filtered_vfat==vfat]
                    if args.verbose: print(f"  chamber {chamber}, eta {eta}, VFAT {vfat}, strips:", filtered_strips_vfat)
                    occupancy_ax.hist(filtered_strips_vfat, bins=358, range=(0,358), label="OH {}, VFAT {}".format(oh, vfat), alpha=0.5)
                occupancy_ax.legend()
                occupancy_ax.set_title("chamber {}, eta {}".format(chamber, eta)) 
                occupancy_ax.set_xlabel("Strip")
                occupancy_fig.savefig(args.odir / f"occupancy_chamber{chamber}_eta{eta}.png")
        
        if args.efficiency: # save efficiency per chamber
            efficiency_df = pd.DataFrame(event_count_chamber, columns=["chamber", "fired"])
            efficiency_df["efficiency"] = efficiency_df["fired"]/n_events
            efficiency_df.to_csv(args.efficiency[1], sep=";", index=False)

if __name__=='__main__': main()
