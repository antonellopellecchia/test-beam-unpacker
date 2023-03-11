> :warning: **This repository is not actively maintained any more**. Please refer to https://gitlab.cern.ch/apellecc/mpgd-analysis.

### Setting up the environment

```bash
git clone git@github.com:antonellopellecchia/testbeam-analysis.git
cd testbeam-analysis
source env.sh
mkdir build && cd build
cmake3 ..
make
```

### Running the code

Running the unpacker:
```bash
RawToDigi raw_file.raw ferol digi.root [n_events]
```

Local reconstruction:
```bash
DigiToRechits digi.raw rechits.root [n_events]
```

Track reconstruction:
```bash
Tracking rechits.raw tracks.root [n_events]
```

Analysis: look at python scripts in the `analysis` folder.

### Utilities:

How to generate a mapping for the tracker:
```bash
python3 mapping/generate.py --connector ../mapping/res/panasonic.csv --geometry ../mapping/july2022/template.csv tracker_mapping.csv
```
where the `geometry` parameter is a csv file containing the mapping from VFAT position to the position in the corresponding chamber and eta partition.
