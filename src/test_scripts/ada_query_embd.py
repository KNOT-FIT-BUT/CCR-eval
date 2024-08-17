
import os
from openai.embeddings_utils import get_embedding, cosine_similarity
import json

api_key = os.getenv("OPENAI_API_KEY")  

with open("../../dareczech/qrel/translated/test_sorted_en.tsv") as qrel:
    with open("qrel_ada.embd.tsv", "w") as out:
        next(qrel)
        prev_query = ""
        embedding = []
        line_num = 0
        for line in qrel:
            print(line_num, end="\r")
            id, query, url, label = line.split("\t")
            if query != prev_query:
                embedding = get_embedding(query, engine="text-embedding-ada-002", api_key=api_key)
            out_format = "\t".join([id, json.dumps(embedding), url, label])
            
            out.write(out_format)
            prev_query = query
            line_num += 1
