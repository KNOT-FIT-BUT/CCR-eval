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
    help="Index type",
    choices=INDEX_TYPES,
    dest="index_type"
)

parser.add_argument(
    "-o", "--output", 
    required=False, 
    action="store", 
    help="Output path for metrics file",
    dest="out_path"
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
    "--grid-search",
    required=False, 
    default=False, 
    action="store_true",
    help="Grid search for different k1, b values (set in config)",
    dest="grid_search"
)

parser.add_argument(
    "--format", 
    required=True, 
    action="store", 
    help="Output format of statistics (graph, table, csv)", 
    choices=["graph", "table", "csv"],
    dest="format"
)

parser.add_argument(
    "--pairs", "--id-url-pairs", 
    required=False, 
    action="store", 
    default=None,
    help="Path to id-url-pairs file (tsv format)", 
    dest="pairs"
)
