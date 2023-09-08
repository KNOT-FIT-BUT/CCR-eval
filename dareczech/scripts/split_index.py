#! /usr/bin/env python3

'''Takes the qrel file, puts the unduplicated urls in a set and 
then creates index slice from docs file with the documents in the set 
+ some random sample   '''

import json
import os

from args.args_split_index import parser

from tqdm import tqdm


args = parser.parse_args()

qrel_file = args.qrel
docs_file = args.docs
index_slice = int(args.slice)
output_docs = args.output

if args.doc_id_key:
    DOC_ID_KEY = args.doc_id_key

if not os.path.exists(qrel_file) or not os.path.exists(docs_file):
    print("Input file(s) not found")

qrel_urls = set()

# Extract ids from qrel file
with open(qrel_file) as qf:
    # Sktip tsv header
    next(qf)

    print("Getting qrel urls...")

    for line in qf:
        id, query, url, label = line.split("\t")
    
        if url not in qrel_urls:
            qrel_urls.add(url)

# Generate slice with 
with open(docs_file) as df, open(output_docs, "w") as of:
    docs_file_lines = sum([1 for _ in open(docs_file)])

    docs_count = 0
    qrel_doc_count = 0

    # Add qrel ids
    for line in tqdm(df, total=docs_file_lines, desc="Getting qrel docs [1/2]", unit="docs"):

        record = json.loads(line)
        url = record[DOC_ID_KEY]

        if url in qrel_urls:
            of.write(json.dumps(record, ensure_ascii=False) + "\n")
            docs_count += 1
            qrel_doc_count += 1
    
    df.seek(0)

    # Add rest of slice (slice)
    total_progress = index_slice - docs_count + qrel_doc_count
    for line in tqdm(df, total=total_progress, desc="Getting other docs [2/2]", unit="docs"):
        if docs_count >= index_slice:
            break 
        
        record = json.loads(line)
        url = record [DOC_ID_KEY]

        if url not in qrel_urls:
            of.write(json.dumps(record, ensure_ascii=False) + "\n")
            docs_count += 1

print("Generated slice of size:", docs_count)
print("\tqrel:", qrel_doc_count)
print("\tother:", docs_count - qrel_doc_count)

if qrel_doc_count < len(qrel_urls):
    print("Warning: not all docs from qrel file were found")
    print("  -> missing:", len(qrel_urls) - qrel_doc_count)
        
    

        






