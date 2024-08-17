import torch
import json
from transformers import AutoModel, AutoTokenizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print("Loading model")
model_name = "Seznam/simcse-dist-mpnet-paracrawl-cs-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

source_docs = "/mnt/minerva1/nlp/projects/CCR/dareczech/documents/sliced/test/100k/dareczech_100k.idnum.tsv"

input_texts = []

print("Loading input texts...")
with open(source_docs, "r") as f:
    for line in f:
       # doc_content = json.loads(line.strip())["doc"]
        _, doc_content = line.split("\t")
        input_texts.append(doc_content)

batch_size = 200
input_batches = [input_texts[i:i+batch_size] for i in range(0, len(input_texts), batch_size)]

# Initialize a list to store all embeddings

# Process each batch
for i, batch in enumerate(input_batches):
    # Tokenize the input texts
    print("Processing batch", i+1, "of", len(input_batches))
    batch_dict = tokenizer(batch, max_length=128, padding=True, truncation=True, return_tensors='pt')
    
    outputs = model(**batch_dict)
    embeddings = outputs.last_hidden_state[:, 0]  # Extract CLS token embeddings
    print("DEBUG: embd.shape", embeddings.shape)

    print("Saving embeddings...")
    torch.save(embeddings, f"/mnt/minerva1/nlp/projects/CCR/indexes/dareczech_simcse-dist-mpnet-paracrawl/cls_embeddings_{i}.pt")

# Combine all embeddings into one tensor
all_embeddings = torch.cat([torch.load(f"/mnt/minerva1/nlp/projects/CCR/indexes/dareczech_simcse-dist-mpnet-paracrawl/cls_embeddings_{i}.pt") for i in range(len(input_batches))])

# Save the combined embeddings
torch.save(all_embeddings, "/mnt/minerva1/nlp/projects/CCR/indexes/dareczech_simcse-dist-mpnet-paracrawl/dareczech-simcse-dist-mpnet-paracrawl_128_noseg.pt")

print("All embeddings saved successfully.")
