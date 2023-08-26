from config import INDEX_TYPES
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--qrel",
    required=True,
    action="store",
    help="Path to input qrel file (.tsv)",
    dest="query_file"
)

parser.add_argument(
    "--index", 
    required=True,
    action="store",
    help="Path to index",
    dest="index_path"
)

parser.add_argument(
    "--index-type", 
    required=True,
    action="store",
    help="Index type (bm25, plaidx)",
    choices=INDEX_TYPES,
    dest="index_type"
)

parser.add_argument(
    "--source-collection", 
    required=False, 
    default="",
    help="Source doc collection (only for colbert)",
    action="store",
    dest="collection"
)

parser.add_argument(
    "--lemmatize-query",
    required=False,
    action="store_true",
    default=False,
    help="Lemmatize query before search",
    dest="lemmatize_query"
)

parser.add_argument(
    "-o", "--output", 
    required=False, 
    action="store", 
    help="Output path for metrics file",
    dest="out_path"
)
