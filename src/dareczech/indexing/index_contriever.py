import torch
import json
from transformers import AutoModel, AutoTokenizer
import re
import os

SPLIT_SENTENCES = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print("Loading model")
tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
model = AutoModel.from_pretrained('facebook/contriever-msmarco').to(device)

out_dir = "/mnt/minerva1/nlp/projects/CCR/indexes/dareczech_100k_en_contriever-msmarco.256.0"

source_docs = "/mnt/minerva1/nlp/projects/CCR/dareczech/documents/sliced/test/100k/dareczech_100k_en.jsonl"
input_texts = []


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1)/mask.sum(dim=1)[..., None]
    return sentence_embeddings
    
print("Loading input texts...")
with open(source_docs, "r") as f:
    doc_sentence_lens = torch.zeros(100000)
    for i, line in enumerate(f):
        doc_content = json.loads(line.strip())["doc"]
        
        if SPLIT_SENTENCES:
            sentences = list(filter(None, re.split(r"\.|!|\?", doc_content)))
            doc_sentence_lens[i] = len(sentences)
            for sentence in sentences:
                input_texts.append(sentence)
        else:
            input_texts.append(doc_content)
        print("Processing doc", i+1, "/100000", end="\r")
    torch.save(doc_sentence_lens, "docs_sentence_lens.pt")
    print()
    
batch_size = 800
print("Passage count:", len(input_texts))
input_batches = [input_texts[i:i+batch_size] for i in range(0, len(input_texts), batch_size)]

# Initialize a list to store all embeddings

# Process each batch
batch_count = len(input_batches)
for i, batch in enumerate(input_batches):

    # Tokenize the input texts
    print("Processing batch", i+1, "of", batch_count)
    batch_dict = tokenizer(batch, max_length=256, padding=True, truncation=True, return_tensors='pt').to(device)
    # 256 tokens in paper (during pre-training)
        
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0] # CLS pooling
    print("DEBUG: embd.shape", embeddings.shape)

    print("Saving embeddings...")
    torch.save(embeddings, os.path.join(out_dir, f"cls_emb_{i}.pt"))
    input_batches = input_batches[1:]
    
del input_batches
del model

# release memory
torch.cuda.empty_cache()


# Combine all embeddings into one tensor memory effiecient
all_embeddings = torch.tensor([]).to(device)
for i in range(batch_count):
    all_embeddings = torch.cat([all_embeddings, torch.load(os.path.join(out_dir, f"cls_emb_{i}.pt")).to(device)])

# 
# Save the combined embeddings
torch.save(all_embeddings, os.path.join(out_dir, "all_cls_embs.pt"))

print("All embeddings saved successfully.")
