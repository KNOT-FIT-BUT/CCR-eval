import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--docs-file", 
    required=True,
    action="store",
    help="Doc file",
    dest="doc_file"
)

parser.add_argument(
    "--out-dir", 
    required=False, 
    action="store", 
    help="Output directory for plots", 
    dest="out_dir"
)
