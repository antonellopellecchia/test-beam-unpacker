import argparse
import pathlib

import numpy as np
import pandas as pd

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=pathlib.Path)
    parser.add_argument("ofile", type=pathlib.Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    template_df = pd.read_csv(args.template)
    if args.verbose: print("Template:\n", template_df)

    vfat_strip_df = pd.read_csv("mapping/vfat.csv")
    if args.verbose: print("VFAT strip mapping:\n", vfat_strip_df)

    n_channels = len(vfat_strip_df.index)
    n_vfats = len(template_df.index)
    print(f"{n_vfats} vfats, {n_channels} channels")

    broadcasted_vfat_df = pd.concat([vfat_strip_df]*n_vfats).reset_index()
    if args.verbose:
        print("Broadcasted VFAT strip mapping:\n", broadcasted_vfat_df)
        print("Size:", len(broadcasted_vfat_df.index))

    template_df = template_df[["vfat", "eta", "position"]]

    mapping_df = pd.DataFrame(np.repeat(template_df.values, n_channels, axis=0))
    mapping_df.columns = template_df.columns
    if args.verbose: print("Broadcasted template:\n", mapping_df)

    mapping_df["channel"] = broadcasted_vfat_df["channel"]
    if args.verbose: print("Adding channel:\n", mapping_df)

    mapping_df["strip"] = n_channels - broadcasted_vfat_df["strip"] + n_channels * mapping_df["position"]
    if args.verbose: print("Adding strips:\n", mapping_df)
    
    mapping_df = mapping_df[["vfat", "channel", "eta", "strip"]]
    mapping_df.columns = ["vfatId", "vfatCh", "iEta", "strip"]
    mapping_df.to_csv(args.ofile, sep=",", index=False)

if __name__=="__main__":
    main()
