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
    "-t", "--extraction-type",
    required=True, 
    action="store",
    choices=["docs", "qrel"],
    help="Extraction type (docs, qrel)",
    dest="extraction_type"
)


parser.add_argument(
    "--output-format", 
    required=True, 
    action="store", 
    choices=["tsv", "jsonl"],
    help="Output format (tsv, jsonl)",
    dest="format"
)
