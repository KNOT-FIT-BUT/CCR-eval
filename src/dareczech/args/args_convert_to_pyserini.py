
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", 
    required=True,
    action="store",
    help="Input file",
    dest="input_file"
)

parser.add_argument(
    "-d", "--out-dir", 
    required=False, 
    action="store", 
    help="Output directory",
    dest="out_dir"
)

parser.add_argument(
    "--lemmatize", 
    required=False, 
    action="store_true", 
    help="Lemmatize tokens (words) in documents",
    dest="lemmatize"
)
