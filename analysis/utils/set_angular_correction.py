import argparse
import pathlib

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("ifile", type=pathlib.Path)
parser.add_argument("ofile", type=pathlib.Path)
parser.add_argument("angle", type=float)
args = parser.parse_args()

geometry_df = pd.read_csv(args.ifile)
geometry_df.loc[geometry_df.chamber==3, "angle"] = args.angle
geometry_df.to_csv(args.ofile, index=None)
print("New geometry file saved to", args.ofile)

