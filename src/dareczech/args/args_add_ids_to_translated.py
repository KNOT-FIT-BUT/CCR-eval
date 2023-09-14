import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--original", 
    required=True,
    action="store",
    help="Original doc file",
    dest="doc_file"
)

parser.add_argument(
    "--translated", 
    required=True, 
    action="store", 
    help="Translated file",
    dest="translated_file"
)

parser.add_argument(
    "-o", "--output", 
    required=True, 
    action="store", 
    help="Output file",
    dest="output_file"
)
