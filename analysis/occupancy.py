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
    parser.add_argument("--start", type=int, default=0, help="First event")
    parser.add_argument("--raw", action="store_true", help="Plot VFAT channel datas")
    parser.add_argument("--find-noisy", type=pathlib.Path, help="Find noisy strips and save them to file")
    parser.add_argument("--mask-noisy", type=pathlib.Path, help="Mask noisy strips found in file")
    parser.add_argument("--efficiency", type=pathlib.Path, nargs="+", help="Save fast efficiency to file")
    parser.add_argument("--latency-cut", type=int, help="Plot separately for different latencies")
    parser.add_argument("--compact", action="store_true", help="Plot all eta with shared x axis")
    parser.add_argument("--check-mapping", action="store_true", help="CSV file for strip mapping to check")
    parser.add_argument("--by-event", type=int, help="Plot strip occupancy vs event number for the first n events")
    parser.add_argument("--ylim", type=int)
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate logging")
    args = parser.parse_args()
    
    os.makedirs(args.odir, exist_ok=True)

    with uproot.open(args.ifile) as digi_file:
        digi_tree = digi_file["outputtree"]
        if args.verbose: digi_tree.show()

        digi_slot = digi_tree["slot"].array(entry_start=args.start,entry_stop=args.start+args.events)
        digi_oh = digi_tree["OH"].array(entry_start=args.start,entry_stop=args.start+args.events)
        digi_vfat = digi_tree["VFAT"].array(entry_start=args.start,entry_stop=args.start+args.events)
        digi_channel = digi_tree["CH"].array(entry_start=args.start,entry_stop=args.start+args.events)
        digi_chamber = digi_tree["digiChamber"].array(entry_start=args.start,entry_stop=args.start+args.events)
        digi_eta = digi_tree["digiEta"].array(entry_start=args.start,entry_stop=args.start+args.events)
        digi_strip = digi_tree["digiStrip"].array(entry_start=args.start,entry_stop=args.start+args.events)

        """for channel in range(128):
            print(channel, ak.count_nonzero(ak.flatten(digi_channel==channel)))"""

        if args.latency_cut:
            latencies = digi_tree["runParameter"].array(entry_start=args.start,entry_stop=args.start+args.events)
            print("Broadcasting latency...")
            digi_latency = latencies * ak.ones_like(digi_channel) # broadcast the latency vector for each channel in event

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

        if args.raw:
            """ Plot occupancy per channel, i.e. before mapping """
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
                        #channel = digi_channel[vfat_filter]
                        """print("Channels:", channel)
                        print("Latencies:", latencies)
                        print("Latencies broadcast:", latencies_channels)"""
                        #latencies_channel = latencies * ak.ones_like(channel) # broadcast the latency vector for each channel in event

                        filtered_channel = ak.flatten(digi_channel[vfat_filter])
                        filtered_oh = ak.flatten(digi_oh[vfat_filter])
                        filtered_vfat = ak.flatten(digi_vfat[vfat_filter])

                        if args.verbose: print(f"slot {slot}, oh {oh}, vfat {vfat}, channels:", filtered_channel)
                        
                        if ak.count(filtered_channel) == 0: continue # no hits for selected oh, vfat
                      
                        if not args.latency_cut:
                            channel_hist, channel_bins, _ = occupancy_axs[vfat].hist(filtered_channel, bins=140, range=(-0.5,139.5))
                        else:
                            nbins = 128*2
                            binrange = (0.5,256.5)
                            filtered_latency = ak.flatten(digi_latency[vfat_filter])
                            channel_hist, channel_bins, _ = occupancy_axs[vfat].hist(
                                    filtered_channel[filtered_latency<args.latency_cut],
                                    bins=nbins, range=binrange, label=f"latency < {args.latency_cut}", alpha=0.4
                            )
                            occupancy_axs[vfat].hist(
                                    filtered_channel[filtered_latency>=args.latency_cut],
                                    bins=nbins, range=binrange, label=f"latency > {args.latency_cut}", alpha=0.4
                            )
                            occupancy_axs[vfat].legend()
                        
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

                for strip in range(0, 385):
                    print(strip, ak.count_nonzero(chamber_strips==strip))

            etas = np.unique(ak.flatten(digi_eta[chamber_filter]))
            if not args.compact: occupancy_fig, occupancy_axs = plt.subplots(figsize=(12*len(etas),10), ncols=len(etas), nrows=1)
            else: occupancy_fig, occupancy_axs = plt.subplots(figsize=(10,6*len(etas)), nrows=len(etas), ncols=1, sharex=True)

            if args.by_event: event_fig, event_axs = plt.subplots(figsize=(11*len(etas),9), ncols=len(etas), nrows=1)

            for ieta,eta in enumerate(etas):

                #if eta>3: break
                occupancy_ax = occupancy_axs[ieta]
                occupancy_ax.set_xlabel("Strip")
                eta_filter = digi_eta==eta
                filtered_strips = ak.flatten(digi_strip[(chamber_filter)&(eta_filter)])
                filtered_oh = ak.flatten(digi_oh[(chamber_filter)&(eta_filter)])
                filtered_vfat = ak.flatten(digi_vfat[(chamber_filter)&(eta_filter)])
                filtered_channels = ak.flatten(digi_channel[(chamber_filter)&(eta_filter)])

                if args.verbose: print(f"chamber {chamber}, eta {eta}, strips:", filtered_strips)
                
                #if ak.count(filtered_strips) == 0: continue # no vfats for selected chamber, eta
               
                oh = np.unique(filtered_oh)[0]
                vfats = np.unique(filtered_vfat)

                if args.check_mapping:
                    """ Print strip, channel and count to check missing channels from mapping """
                    for strip in range(256):
                        strip_filter = filtered_strips==strip
                        strip_occupancy = ak.count_nonzero(strip_filter, axis=0)
                        channels_for_strip = filtered_channels[strip_filter]
                        channel, err_channel = ak.mean(channels_for_strip), ak.std(channels_for_strip)
                        if strip_occupancy == 0 or err_channel !=0:
                            print(f"Found OH {oh}, chamber {chamber}, eta {eta}, strip {strip}: channel {channel} ± {err_channel:1.2f}, occupancy {strip_occupancy}")
                            if err_channel > 0:
                                # multiple channels for a single strip, print them to check what's happening...
                                print(f"Multiple channels for strip {strip}:", np.unique(channels_for_strip))

                    """ Do the same in a channel-by-channel fashion """
                    for vfat in vfats:
                        for channel in range(128):
                            filtered_channels_vfat = filtered_channels[filtered_vfat==vfat]
                            filtered_strips_vfat = filtered_strips[filtered_vfat==vfat]
                            
                            channel_filter = filtered_channels_vfat==channel
                            channel_occupancy = ak.count_nonzero(channel_filter, axis=0)
                            strips_for_channel = filtered_strips_vfat[channel_filter]
                            strip, err_strip = ak.mean(strips_for_channel), ak.std(strips_for_channel)

                            if channel_occupancy == 0 or err_strip !=0:
                                print(f"Found OH {oh}, chamber {chamber}, eta {eta}, channel: {channel}: strip {strip} ± {err_strip:1.2f}, occupancy {channel_occupancy}")
                                if err_strip > 0: print(f"Multiple strips for channel {channel}:", np.unique(strips_for_channel))
                            elif args.verbose:
                                print(f"Found OH {oh}, chamber {chamber}, eta {eta}, channel: {channel}: strip {strip} ± {err_strip:1.2f}, occupancy {channel_occupancy}")

                if args.by_event:
                    n_events = args.by_event
                    event_ax = event_axs[ieta]

                    forplot_strips = digi_strip[(chamber_filter)&(eta_filter)][:n_events]
                    n_strips = ak.max(forplot_strips)
                    strip_range = np.arange(n_strips)
                    event_range = np.arange(len(forplot_strips))
                    event_broad, _ = ak.broadcast_arrays(event_range, forplot_strips)
                    if args.verbose:
                        print(f"{n_events} events, {n_strips} strips")
                        print("Broadcasted index:", ak.flatten(event_broad))
                        print("Strips:", ak.flatten(forplot_strips))
                    event_ax.hist2d(
                        ak.flatten(forplot_strips, axis=None).to_numpy(),
                        ak.flatten(event_broad, axis=None).to_numpy(),
                        bins=(n_strips,n_events),
                        range=[(0.5,n_strips+0.5),(0.5,n_events+0.5)]
                    )
                    event_ax.set_xlabel("Strip")
                    event_ax.set_ylabel("Event number")

                nbins = 128*3
                binrange = (0.5,384.5)
 
                if not args.latency_cut:
                    if args.compact:
                        occupancy_ax.margins(y=0)
                        occupancy_ax.text(.75, .75, f"iη={eta}", fontweight="bold", fontsize=25, transform=occupancy_ax.transAxes)
                        if ak.count(filtered_strips)==0:
                            if args.verbose: print(f"Deleting axes chamber {chamber} eta {eta}")
                            occupancy_fig.delaxes(occupancy_ax)
                            continue
                        occupancy_ax.hist(filtered_strips, bins=nbins, range=binrange, alpha=0.3, color=["blue","purple","red","yellow"][ieta])
                    else:
                        for vfat in vfats:
                            filtered_strips_vfat = filtered_strips[filtered_vfat==vfat]
                            if args.verbose: print(f"  chamber {chamber}, eta {eta}, VFAT {vfat}, strips:", filtered_strips_vfat)
                            occupancy_ax.hist(filtered_strips_vfat, bins=nbins, range=binrange, label="OH {}, VFAT {}".format(oh, vfat), alpha=0.5)

                else:
                    """ Plot separately occupancy for different latency peaks
                    Useful for cross-talk in latency scans """
                    filtered_latency = ak.flatten(digi_latency[(chamber_filter)&(eta_filter)])
                    if args.verbose: print(f"  chamber {chamber}, eta {eta}, VFAT {vfat}, strips:", filtered_strips)
                    occupancy_ax.hist(filtered_strips[filtered_latency<args.latency_cut], bins=nbins, range=binrange, label=f"latency < {args.latency_cut}", alpha=0.4)
                    occupancy_ax.hist(filtered_strips[filtered_latency>=args.latency_cut], bins=nbins, range=binrange, label=f"latency > {args.latency_cut}", alpha=0.4)

                if args.ylim: occupancy_ax.set_ylim(0, args.ylim)
                if not args.compact:
                    occupancy_ax.legend()
                    occupancy_ax.set_title("chamber {}, eta {}".format(chamber, eta)) 

            occupancy_fig.savefig(args.odir / f"occupancy_chamber{chamber}.png")
            if args.by_event: event_fig.savefig(args.odir / f"events_chamber{chamber}.png")
        
        if args.efficiency: # save efficiency per chamber
            efficiency_df = pd.DataFrame(event_count_chamber, columns=["chamber", "fired"])
            efficiency_df["efficiency"] = efficiency_df["fired"]/n_events
            efficiency_df.to_csv(args.efficiency[1], sep=";", index=False)

if __name__=='__main__': main()
