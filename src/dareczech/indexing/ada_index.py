import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding
import json

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
encoding = tiktoken.get_encoding(embedding_encoding)


data = []
print("Loading dareczech docs.. ", end="")
with open("/home/xsteti05/project/dareczech/documents/sliced/test/100k/dareczech_100k_en.jsonl") as file:
    for i,line in enumerate(file):
        data.append(list(json.loads(line).values()))
print("done.")

print("Initializing dataframe.. ", end="")
df  = pd.DataFrame(data, columns=["url", "doc"])
print("done.")


print("Getting number of tokens.. ", end="")
df["n_tokens"] = df["doc"].apply(lambda x: len(encoding.encode(x)))
print("done.")

print("Truncating long docs.. ", end="")
df.loc[df["n_tokens"] > max_tokens, "doc"] = df.loc[df["n_tokens"] > max_tokens, "doc"].apply(lambda x: encoding.decode(encoding.encode(x)[:max_tokens]))
df["n_tokens"] = df["doc"].apply(lambda x: len(encoding.encode(x)))
print("done.")

doc_count = 0
err_count = 0
print("Embedding...")
df["embedding"] = None
for index, row in df.iterrows():
    print("docs:", doc_count, end="\r")
    url = str(row["url"])
    doc = str(row["doc"]).replace("\n", " ")
    
    if doc == "":
        doc = " "

    try:
        df.at[index, "embedding"] = get_embedding(doc, engine=embedding_model, api_key="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")
    except:
        err_count += 1

    doc_count += 1
print()
print("Errors count: ", err_count)
print("Finished, saving...")
df.to_pickle("dareczech_100k_en_ada.embed.8000.pickle")
df.to_csv("dareczech_100k_en_ada.embed.8000.csv", index=False)
print("Saved.")