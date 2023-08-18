#! /bin/bash

INDEX_PATH="../../indexes/dareczech_bm25"
DOCUMENT_COLLECTION_PATH="../dareczech/documents/lemmatized"

mkdir -p "$DOCUMENT_COLLECTION_PATH"

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --language cs \
  --input "$DOCUMENT_COLLECTION_PATH" \
  --index "$INDEX_PATH" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
