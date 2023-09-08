#! /usr/bin/env python3

from args.args_split_docs import parser, test_text
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os

args = parser.parse_args()

DOC_ID_KEY = args.doc_id_key
input_docs = args.input_docs
output_docs = args.output_docs
split_threshold = int(args.split_threshold)
overlap = int(args.overlap)
tokenizer = args.tokenizer


if split_threshold <= 0:
    print("Invalid split threshold value:", split_threshold)
    exit(1)

if overlap >= split_threshold:
    print("Overlap too big")
    exit(1)


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer)
TOKENIZER_MAX_SEQ_LENGHT = tokenizer.model_max_length

with open(input_docs) as docs_in, open(output_docs, "w") as docs_out:
    lines_num = sum([1 for _ in open(input_docs)])

    for line in tqdm(docs_in, total=lines_num):
        line = json.loads(line)
        doc_id = line[DOC_ID_KEY]
        doc_content = line["doc"]
        tokens = tokenizer.tokenize(doc_content)
        
        # Split threshold not exceeded
        if len(tokens) <= split_threshold:
            docs_out.write(json.dumps(line, ensure_ascii=False) + "\n")

        else:
            start_idx = 0
            doc_idx = 1

            index_limit = len(tokens) if len(tokens) < TOKENIZER_MAX_SEQ_LENGHT else TOKENIZER_MAX_SEQ_LENGHT
            while start_idx < index_limit:
                end_idx = min(start_idx + split_threshold, len(tokens))
                doc_content = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx])
                docs_out.write(json.dumps({
                    DOC_ID_KEY:f"{doc_id}|{doc_idx}", 
                    "doc":doc_content
                }, ensure_ascii=False) + "\n")
                start_idx += split_threshold - overlap
                doc_idx += 1
