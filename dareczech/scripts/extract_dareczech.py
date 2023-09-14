#! /usr/bin/python3

from tqdm import tqdm
import json
import re  

from args.args_extract_dareczech import parser
from utils import squash_doc_content, lines_in_file
from dareczech_reg import DOC_BTE_REG

args = parser.parse_args()

input_file = args.input_file
doc_file  = args.output_file
out_file_format = args.format
extraction_type = args.extraction_type

print("Extraction type:", extraction_type)
print("Output format: ", out_file_format)

with open(input_file) as file_in, open(doc_file, "w") as doc_file:
    # Skip tsv header
    next(file_in)

    saved_docs = set()
    duplicates_count = 0
    lines_count = lines_in_file(input_file) - 1
    line_idx = 0

    for line in tqdm(file_in, total=lines_count, desc="Processing", unit="docs"):
        fields = line.strip().split('\t')
        id, query, url, doc, title, label = fields
                
        doc_bte = re.search(DOC_BTE_REG, str(doc)).group(1)
        doc_content = squash_doc_content(title=title, doc_bte=doc_bte)
        
        if extraction_type == "docs":
            # Remove duplicates
            if url in saved_docs:
                duplicates_count += 1
                continue
            saved_docs.add(url)

            if out_file_format == "jsonl":
                out_format = {
                    "url": url,
                    "doc": doc_content
                }
                data_out = json.dumps(out_format, ensure_ascii=False)
            
            elif out_file_format == "tsv":
                doc_content = doc_content.replace("\n", " ")
                data_out = f"{line_idx}\t{doc_content}"
        
        elif extraction_type == "qrel":
             # Remove duplicates
            if query in saved_docs:
                duplicates_count += 1
                continue
            saved_docs.add(query)

            if out_file_format == "jsonl":
                out_format = {
                    "doc": query
                }
                data_out = json.dumps(out_format, ensure_ascii=False)
            
            elif out_file_format == "tsv":
                data_out = f"{line_idx}\t{query}"

        doc_file.write(data_out + "\n")
        line_idx += 1 

print("Duplicates:", duplicates_count)
print("Done.")

