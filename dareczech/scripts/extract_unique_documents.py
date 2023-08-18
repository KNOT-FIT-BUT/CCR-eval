#!/usr/bin/python3

import re  
import json

from dareczech_reg import DOC_BTE_REG

input_file = "../raw/dareczech.tsv"
doc_file = "../documents/dareczech_docs.jsonl"

def squash_doc_content(
        title:str,
        doc_title:str="",
        doc_bte:str="", 
        sep:str="\n") -> str:
    return title + sep + doc_title + sep + doc_bte + sep


with open(input_file) as file_in, open(doc_file, "w") as doc_file:
    # Skip tsv header
    next(file_in)

    saved_docs = set()
    duplicates_count = 0
    for line in file_in:
        fields = line.strip().split('\t')
        id, query, url, doc, title, label = fields
        
        if url in saved_docs:
            duplicates_count += 1
            continue
        
        saved_docs.add(url)
        
        doc_bte = re.search(DOC_BTE_REG, str(doc)).group(1)
        doc_content = squash_doc_content(title=title, doc_bte=doc_bte)
        
        out_format = {
            "url": url,
            "doc": doc_content
        }

        doc_file.write(json.dumps(out_format, ensure_ascii=False) + "\n")

print("Duplicates:", duplicates_count)

