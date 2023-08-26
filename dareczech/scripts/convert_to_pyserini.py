#! /usr/bin/env python3

"""
Author: Jakub Stetina
Email: xsteti05@stud.fit.vutbr.cz
Description: This Python script is used for converting document files from the 'dareczech' dataset
             into a pyserini (indexer) compatible .jsonl format 
Usage:                                       
    dareczech.py [-h] -i INPUT_FILE [-d OUT_DIR]

Arguments:
    -i, --input: Path to an input .jsonl format with 'url' and 'doc' fields on each line
    -d, --out-dir: (optional) Path to an output directory 

Note:   Output file name will be the same as the input file with the new .jsonl extension,
        output files with the same name will be overwritten! 
"""

from tqdm import tqdm
import logging
import json
import os

from lemmatize import Lemmatizer
from lemmatize import ModelLoadError, ModelNotLoadedError, TokenizerError

from args.args_convert_to_pyserini.py import parser

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Default out directory
OUT_DIR = "../documents"

args = parser.parse_args()

if args.lemmatize:
    OUT_DIR = "../documents/converted/lemmatized"

input_path = os.path.abspath(args.input_file)
if not os.path.exists(input_path):
    logging.error(f"Input file does not exist ({input_path})")
    exit(1)

logging.info("Starting...")
input_lines_num = sum(1 for line in open(input_path)) - 1

if args.out_dir:
    OUT_DIR = args.out_dir

if not input_path.endswith(".jsonl"):
    logging.warning("Input file might not be in the correct format (expected .jsonl)")

output_path = os.path.basename(os.path.abspath(input_path))
output_path = f"{OUT_DIR}/{os.path.splitext(output_path)[0]}.jsonl"

if args.lemmatize:
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
    for line in tqdm(file_in, total=input_lines_num, desc="Processing", unit="line"):
        data = json.loads(line)
        
        url = data['url']
        doc = data['doc']
        
        if args.lemmatize:
            doc = lemmatizer.lemmatize_text(doc)

        # Pyserini expected format
        doc_json = {
            "id": url,
            "contents" : doc
        }

        file_out.write(json.dumps(doc_json, ensure_ascii=False) + "\n")

logging.info("Done")
