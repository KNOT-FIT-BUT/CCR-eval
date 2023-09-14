import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--in-dir", 
    required=True,
    action="store",
    help="Dareczech dir",
    dest="in_dir"
)

parser.add_argument(
    "-o", "--out-dir", 
    required=True, 
    action="store", 
    help="Output directory",
    dest="out_dir"
)
