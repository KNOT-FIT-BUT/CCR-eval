import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", 
    required=True,
    action="store",
    help="Input file .tsv file",
    dest="input_file"
)

parser.add_argument(
    "-o", "--output", 
    required=True, 
    action="store", 
    help="Output doc file",
    dest="output_file"
)

parser.add_argument(
    "--format", 
    required=False, 
    action="store", 
    choices=["tsv", "jsonl"],
    help="Output format (.tsv, .jsonl)",
    default="jsonl",
    dest="format"
)
