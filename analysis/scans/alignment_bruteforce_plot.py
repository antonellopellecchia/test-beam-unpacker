import argparse
import pathlib

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("ifile", type=pathlib.Path)
parser.add_argument("ofile", type=pathlib.Path)
args = parser.parse_args()

angle_df = pd.read_csv(args.ifile, sep=";")
print(angle_df)

fig, ax = plt.figure(figsize=(11,9)), plt.axes()
ax.errorbar(angle_df.angle, angle_df.space_resolution, yerr=angle_df.err_space_resolution, fmt="o")
fig.savefig(args.ofile)

