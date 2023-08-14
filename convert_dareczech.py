#! /usr/bin/python3

"""
Author: Jakub Stetina
Email: xsteti05@stud.fit.vutbr.cz
Description: This Python script is used for converting files from the 'dareczech' dataset
             into a JSONL format. It removes duplicates and performs document stemming
             and lemmatization for each token.
Usage:                                       
    dareczech.py [-h] -i INPUT_FILE [-d OUT_DIR]

Arguments:
    -i, --input: Path to an input .tsv file from dareczech dataset
    -d, --out-dir: (optional) Path to an output directory 

Note:   Output file name will be the same as the input file with the new .jsonl extension,
        output files with the same name will be overwritten! 
"""

import argparse
import logging
import json
import os
import re
from tqdm import tqdm

from lemmatize import Lemmatizer
from lemmatize import ModelLoadError, ModelNotLoadedError, TokenizerError
from dareczech_reg import DOC_TITLE_REG, DOC_URL_REG, DOC_BTE_REG

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def squash_doc_content(
        title:str,
        doc_title:str="",
        doc_bte:str="", 
        sep:str="\n") -> str:
    return title + sep + doc_title + sep + doc_bte + sep

# Default out directory
OUT_DIR = "dareczech/lemmatized"

# Parse arguments
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

args = parser.parse_args()
input_path = os.path.abspath(args.input_file)
if not os.path.exists(input_path):
    logging.error(f"Input file does not exist ({input_path})")
    exit(1)

logging.info("Starting...")
input_lines_num = sum(1 for line in open(input_path)) - 1

if args.out_dir:
    OUT_DIR = args.out_dir


if not input_path.endswith(".tsv"):
    logging.warning("Input file might not be in the correct format (expected .tsv)")

output_path = os.path.basename(os.path.abspath(input_path))
output_path = f"{OUT_DIR}/{os.path.splitext(output_path)[0]}.jsonl"

lemmatizer = Lemmatizer()

try:
    logging.info("Loading model...")
    lemmatizer.load_model()
except FileNotFoundError:
    logging.error("Morpho model not found.")
    exit(1)
except ModelLoadError:
    logging.error("Error while loading czech morpho model.")
    exit(1)

logging.info("Model loaded successfully.")

with open(input_path) as file_in, open(output_path, "w") as file_out:
    # Skip first line (tsv header)
    next(file_in)

    converted_docs = set()
    skip_count = 0

    for line in tqdm(file_in, total=input_lines_num, desc="Processing", unit="line"):
        data = line.split("\t")
        
        id      = data[0]
        # query   = data[1]
        url     = data[2]
        doc     = data[3]
        title   = data[4]
        # label   = data[5]

        
        if url in converted_docs:
            skip_count += 1
            continue

        converted_docs.add(url)
        
        doc_title   = re.search(DOC_TITLE_REG, doc).group(1)
        # doc_url     = re.search(DOC_URL_REG, doc).group(1)
        doc_bte     = re.search(DOC_BTE_REG, doc).group(1)
        doc_bte = lemmatizer.lemmatize_text(doc_bte)

        doc_content_squash = squash_doc_content(title=title, doc_title=doc_title, doc_bte=doc_bte)

        # Pyserini expected format
        doc_json = {
            "id": id,
            "contents" : doc_content_squash
        }

        file_out.write(json.dumps(doc_json, ensure_ascii=False) + "\n")

logging.info("Done")
logging.info(f"Duplicate documents: {skip_count}")
