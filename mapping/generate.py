import argparse
import pathlib

import numpy as np
import pandas as pd

verbose = False

def pin_to_channel(connector_type, pin):

    if connector_type == "panasonic":
        channel = (pin%2==1)*((pin-1)%4==2)*(pin-3)
        channel += (pin%2==1)*((pin-1)%4==0)*(pin-4)
        channel += (pin%2==0)*(pin%4==2)*pin
        channel += (pin%2==0)*(pin%4==0)*(pin-1)
        return channel
    else:
        raise ValueError("Connector type {} not recognized".format(connector_type))

def mapping_connector(connector, template_df):
    """ Generate mapping using connector geometry """
    
    vfat_strip_df = pd.read_csv(connector)
    vfat_strip_df["strip"] = vfat_strip_df["pin"]-2
    if verbose: print("VFAT strip mapping:\n", vfat_strip_df)

    n_channels = len(vfat_strip_df.index)
    n_vfats = len(template_df.index)
    print(f"{n_vfats} vfats, {n_channels} channels")

    broadcasted_vfat_df = pd.concat([vfat_strip_df]*n_vfats).reset_index()
    if verbose:
        print("Broadcasted VFAT strip mapping:\n", broadcasted_vfat_df)
        print("Size:", len(broadcasted_vfat_df.index))

    template_df = template_df[["vfat", "eta", "position"]]

    mapping_df = pd.DataFrame(np.repeat(template_df.values, n_channels, axis=0))
    mapping_df.columns = template_df.columns
    if verbose: print("Broadcasted template:\n", mapping_df)

    mapping_df["channel"] = broadcasted_vfat_df["channel"]
    if verbose: print("Adding channel:\n", mapping_df)

    mapping_df["strip"] = n_channels-1 - broadcasted_vfat_df["strip"] + n_channels * mapping_df["position"]
    if verbose: print("Adding strips:\n", mapping_df)
    
    mapping_df = mapping_df[["vfat", "channel", "eta", "strip"]]
    mapping_df.columns = ["vfatId", "vfatCh", "iEta", "strip"]
    return mapping_df

def mapping_analytical(template_df):
    """ Generate mapping using geometry and analytical Panasonic 130 mapping """

    pins = np.arange(2,130)
    channels = pin_to_channel("panasonic", pins)
    local_strips = pins - 2
    vfat_mapping_df = pd.DataFrame({"channel": channels, "localStrip": local_strips})
    vfat_mapping_df.set_index("channel", inplace=True)

    n_vfats = len(template_df.index)
    n_channels = len(channels)
    print(f"{n_vfats} vfats, {n_channels} channels in geometry")
    #template_df = template_df[["vfat", "eta", "position"]]

    mapping_df = pd.concat([
        template_df.assign(channel=ch) for ch in channels
        #template_df.assign(channel=ch).assign(localStrip=local_strips[channels==ch]) for ch in channels
    ])
    #mapping_df["localStrip"] = vfat_mapping_df[vfat_mapping_df["channel"]==mapping_df["channel"]]["localStrip"]
    mapping_df = mapping_df.join(vfat_mapping_df, on="channel")
    mapping_df["strip"] = n_channels - mapping_df["localStrip"] + n_channels*mapping_df["position"]

    # keep only columns we need for unpacking:
    mapping_df = mapping_df[["vfat", "channel", "eta", "strip"]]
    # use unpacker naming conventions:
    mapping_df.columns = ["vfatId", "vfatCh", "iEta", "strip"]

    if verbose:
        print("Pins:", pins)
        print("Channels:", channels)
        print("VFAT mapping:\n", vfat_mapping_df)
        print("Mapping dataframe:\n", mapping_df)

    return mapping_df

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--connector", type=pathlib.Path)
    parser.add_argument("--geometry", type=pathlib.Path)
    parser.add_argument("ofile", type=pathlib.Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    global verbose
    verbose = args.verbose

    template_df = pd.read_csv(args.geometry)
    if verbose: print("Template:\n", template_df)

    if not args.connector:
        print("No connector mapping provided, using analytical mapping for Panasonic 130...")
        mapping_df = mapping_analytical(template_df)
    else:
        print("Using connector mapping in", args.connector)
        mapping_df = mapping_connector(args.connector, template_df)
    
    mapping_df.to_csv(args.ofile, sep=",", index=False)
    print("Saved mapping file to", args.ofile)

if __name__=="__main__":
    main()
