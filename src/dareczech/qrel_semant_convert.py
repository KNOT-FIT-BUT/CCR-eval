#! /usr/bin/env python3 

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", 
    required=True, 
    action="store", 
    help="Input tsv qrel file",
    dest="qrel_in"
)

parser.add_argument(
    "-o", "--output", 
    required=True, 
    action="store", 
    help="Output file (tsv/jsonl depending on --conversion arg",
    dest="qrel_out"
)

parser.add_argument(
    "--conversion",
    required=True, 
    action="store",
    choices=["after-translation", "before-translation"], 
    help='''Output format (tsv, jsonl)\ntsv=qrel format\njsonl=semAnt format''',
    dest="conversion"
)

parser.add_argument(
    "--translated-queries",
    required=False, 
    default=None,
    action="store", 
    dest="qrel_trans"
)

args = parser.parse_args()

input_path = args.qrel_in
output_path = args.qrel_out
qrel_trans = args.qrel_trans
conversion_type = args.conversion

# jsonl format -- before translation 
if conversion_type == "before-translation":
    with open(input_path) as tsv_in, open(output_path, "w") as jsonl_out:
        next(tsv_in)
        for line in tsv_in:
            id, query, url, label = line.split("\t")
            jsonl_out.write(json.dumps({"text":query}, ensure_ascii=False) + "\n")


# Tsv format -- after translation 
elif conversion_type == "after-translation":
    if qrel_trans is None:
        print("Error: --translated-queries arg expected")
        exit(1)

    with open(input_path) as tsv_in, open(qrel_trans) as jsonl_in, open(output_path, "w") as tsv_out:
        tsv_out.write(tsv_in.readline())
        for jsonl_in_line, tsv_in_line in zip(jsonl_in, tsv_in):
            id, query, url, label = tsv_in_line.split("\t")
            translated_query = json.loads(jsonl_in_line)["translation"]
            tsv_out.write(f"{id}\t{translated_query}\t{url}\t{label.rstrip()}\n")
    
    # os.remove(input_path)



