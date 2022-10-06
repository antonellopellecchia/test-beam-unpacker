import sys
import argparse
import pathlib
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ifile", type=pathlib.Path, help="Input file")
    parser.add_argument("value", type=str, help="x or y or angle")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    corrections_df = pd.read_csv(args.ifile, sep=";")
    if args.verbose: print(corrections_df)

    if args.value=="x": corrections = corrections_df["translation_x"]
    elif args.value=="y": corrections = corrections_df["translation_y"]
    elif args.value=="angle": corrections = corrections_df["angle"]
    else: return -1

    #if args.value=="angle": corrections *= -1
    print(" ".join(corrections.astype(str)))
    return 0

if __name__=="__main__": sys.exit(main())
