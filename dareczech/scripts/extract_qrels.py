#!/usr/bin/python3

import re  
import argparse
import os

from args.args_extract_qrels import parser

QREL_HEADER = "id\tquery\turl\tlabel"

args = parser.parse_args()

if args.out_dir == args.in_dir:
    print("Error: Input and output paths are the same (this would rewrite the files)")
    exit(1)

in_dir = args.in_dir.rstrip("/")
out_dir = args.out_dir.rstrip("/")

dareczech_files = ["dev.tsv", "test.tsv", "train_small.tsv", "train_big.tsv"]

for file in dareczech_files:
    print(f"Current file: {in_dir}/{file}")

    qrel_path = f"{out_dir}/{file}"

    with open(f"{in_dir}/{file}") as file_in, open(qrel_path, "w") as qrel_file:
        # Skip tsv header
        next(file_in)

        # Write new header
        qrel_file.write(f"{QREL_HEADER}")

        for line in file_in:
            fields = line.strip().split('\t')
            id, query, url, doc, title, label = fields
            
            qrel_file.write(f"\n{id}\t{query}\t{url}\t{label}")

    print(f"Saving to: {qrel_path}")

