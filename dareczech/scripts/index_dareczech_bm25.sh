#! /bin/bash

INDEX_PATH="../../indexes/dareczech_bm25"
DOCUMENT_COLLECTION_PATH="../dareczech/documents/lemmatized"


display_usage() {
    echo "Usage: $0 [-i INDEX_PATH] [-c DOCUMENT_COLLECTION_PATH]"
    echo "Options:"
    echo "  -i,                Path to the index directory (optional)"
    echo "  -c,                Path to the document collection directory (optional)"
    echo "  -h,                Display this help message"
}

# Parse args
while getopts ":i:c:h" opt; do
    case $opt in
        i)
            INDEX_PATH="$OPTARG"
            ;;
        c)
            DOCUMENT_COLLECTION_PATH="$OPTARG"
            ;;
        h)
            display_usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            display_usage
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument."
            display_usage
            exit 1
            ;;
    esac
done

# Display the values of the arguments
echo "Index Path: $INDEX_PATH"
echo "Document Collection Path: $DOCUMENT_COLLECTION_PATH"

mkdir -p "$INDEX_PATH"

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --language cs \
  --input "$DOCUMENT_COLLECTION_PATH" \
  --index "$INDEX_PATH" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
