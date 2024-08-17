#! /usr/bin/env python3 
import json
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
        "-i", "--input",
        type=str, 
        required=True,
        dest="input"
)

parser.add_argument(
        "--out-dir", 
        type=str, 
        required=False,
        dest="out_dir"
)

args = parser.parse_args()

input_path = args.input
input_basename = os.path.splitext(os.path.basename(input_path))[0]


output_dir = os.path.dirname(input_path) if not args.out_dir else args.out_dir                                                    

pairs_out_path = f"{output_dir}/{input_basename}.pairs.tsv"
docs_out_path = f"{output_dir}/{input_basename}.idnum.tsv"


with open(input_path) as docs_in:
    with open(docs_out_path, "w") as docs_out:
        with open(pairs_out_path, "w") as pairs_out:
            line_idx = 0
            for line in docs_in:    
                    line = json.loads(line)
                    doc_url = line['url']
                    doc_content = line['doc'].replace("\n", " ")
                    docs_out.write(f"{line_idx}\t{doc_content}\n")
                    pairs_out.write(f"{line_idx}\t{doc_url}\n")
                    line_idx += 1 
                    
