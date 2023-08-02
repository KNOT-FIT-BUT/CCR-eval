# CzechDocLemm-BM25

Detailed documentation for this project can be found [here](https://knot.fit.vutbr.cz/wiki/index.php/CCR).

## Short description
Process, analyze and index a dataset of documents in the Czech language. The core functionality of this project centers around the BM25 model, a powerful ranking function commonly used in information retrieval systems. The BM25 model enables efficient indexing and scoring of the lemmatized documents, facilitating fast and accurate retrieval of relevant documents based on user queries.

## Key Features
1. **Czech Document Lemmatization:** The repository provides robust lemmatization capabilities tailored specifically for the Czech language using the MorphoDiTa model.

2. **BM25 Indexing:** The project implements the BM25 ranking model to create an index of the lemmatized documents. This index significantly improves search performance and precision when querying the dataset.

## Scripts

### darezech.py
Lemmatizes .tsv documents from the dareczech corpora using MorphoDiTa and converts them to a .jsonl format. 

### fetch_czech_morph_model.sh
Gets the latest morphological model (required by ufal.morphodita, used for docuemnt lemmatization) and stores it in models/ directory.


