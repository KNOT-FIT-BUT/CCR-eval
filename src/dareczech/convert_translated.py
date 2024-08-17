import json
with open("../documents/translated/dareczech_docs_en.v2.ids.jsonl") as file:
    with open("../documents/translated/dareczech_docs_en.v2.tsv.testing", "w") as tsv_out:
        with open("../documents/translated/dareczech_docs_en.v2.url-id-pairs.tsv", "w") as pairs_out:
            line_idx = 0
            for line in file:
                line = json.loads(line)
                doc_content = line["doc"].replace('\n', ' ')
                url = line["url"]
                tsv_out.write(f"{line_idx}\t{doc_content}\n")
                pairs_out.write(f"{line_idx}\t{url}\n")
                line_idx += 1
