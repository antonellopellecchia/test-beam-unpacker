import argparse
import pathlib

import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry", type=str, help="Geometry (e.g. may2022)")
    parser.add_argument("chamber", type=int, help="Chamber to apply correction to")
    parser.add_argument("parameter", type=str, help="x, y, z, or angle")
    parser.add_argument("correction", type=float, help="")
    parser.add_argument("-v", "--verbose", action="store_true", help="Activate logging")
    args = parser.parse_args()

    geometry_dir = pathlib.Path("geometry/")
    geometry_csv = geometry_dir / f"{args.geometry}.csv"

    geometry_df = pd.read_csv(geometry_csv, index_col="chamber")
    print("Old geometry:\n", geometry_df)
    print("Applying correction", args.correction, "to parameter", args.parameter, "for chamber", args.chamber)
    geometry_df.at[args.chamber, args.parameter] += args.correction

    print("New geometry:\n", geometry_df)
    geometry_df.to_csv(geometry_csv)
    print("New geometry saved to", geometry_csv)

if __name__=="__main__":
    main()
