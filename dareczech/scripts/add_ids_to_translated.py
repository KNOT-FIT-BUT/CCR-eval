#! /usr/bin/env python3

import json

from args.args_add_ids_to_translated import parser

args = parser.parse_args()

doc_file = args.doc_file
trans_file = args.translated_file
out_file = args.output_file

docs_count = 0
with open(doc_file) as df, open(trans_file) as tf, open(out_file, "w") as of:
    for doc_line, trans_line in zip(df, tf):
        docs_count += 1
        print(f"Processing, docs count: {docs_count}", end="\r")
        doc_url = json.loads(doc_line)["url"]
        trans_line = json.loads(trans_line)
        trans_line = {"url": doc_url, **trans_line}

        of.write(json.dumps(trans_line, ensure_ascii=False) + "\n")
    
    print()
    print("Done.")



