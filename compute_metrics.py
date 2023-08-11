#! /usr/bin/python3

from tqdm import tqdm
import argparse
import logging
import sys
import os

from pyserini.search.lucene import LuceneSearcher
from lemmatize import ModelLoadError, ModelNotLoadedError, TokenizerError
from lemmatize import Lemmatizer

from time import sleep
from datetime import datetime

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Anything with score above 0 considered relevant
RELEVANCE_THRESHOLD = 0.0 

# Count metrics for first k result
METRICS_AT_K = [3,5,7,10]

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--query-file",
    required=True,
    action="store",
    help="Path to input query file (.tsv)",
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
    "-o", "--output", 
    required=False, 
    action="store", 
    help="Output path for metrics file",
    dest="out_path"
)

args = parser.parse_args()

query_file:str= args.query_file
index_file:str = args.index_path
out_path:str = args.out_path

# Perform i/o file checks
if not os.path.exists(query_file):
    logging.error("Query path does not exist, stopping..")
    exit(1)
elif query_file.endswith(".tsv"):
    logging.warning("Query file might not be in the correct format")

if not os.path.exists(index_file):
    logging.error("Index path does not exist, stopping..")
    exit(1)

if not out_path:
    out_file = sys.stdout

else:
    out_file = open(out_path, "w")


logging.info("Loading index..")
searcher = LuceneSearcher(index_file)
logging.info("Index loaded.")


logging.info("Loading morpho model...")
try:
    lemmatizer = Lemmatizer()
    lemmatizer.load_model()
except FileNotFoundError:
    logging.error("Morpho model not found.")
    exit(1)
except ModelLoadError:
    logging.error("Error while loading czech morpho model.")
    exit(1)
logging.info("Model loaded.")

lines_count = sum(1 for line in open(query_file)) - 1
with open(query_file) as q_file:

    # Skip query file header
    next(q_file)

    current_query = ""
    document_relevance = {}
    queries_count = 0

    # Metrics
    relevant_count = 0

    precisions_at = {}
    recalls_at = {}
    mrr_at = {}
    map_at = {}

    # Initialize dictionaries
    for k in METRICS_AT_K:
        precisions_at[k] = 0.0
        recalls_at[k] = 0.0
        mrr_at[k] = 0.0
        map_at[k] = 0.0

    for line in tqdm(q_file, total=lines_count, desc="Computing", unit="queries"):
        data = line.split("\t")

        id      = data[0]
        query   = data[1]
        label   = float(data[5])

        # New test-query
        if query != current_query:

            # Perform current query search
            if current_query:
                queries_count += 1

                results = searcher.search(lemmatizer.lemmatize_text(current_query))
                total_relevant = sum(1 for value in document_relevance.values() if value is True)

                running_percision = 0                

                for i, result in enumerate(results):
                    result_id = result.docid

                    # Retrieved document is relevant
                    if document_relevance.get(result_id) is True:
                        relevant_count += 1

                        # If first relevant document
                        if relevant_count == 1:
                            first_relevant_rank = (i+1)
                    
                    running_percision += relevant_count/(i+1)

                    for k in METRICS_AT_K:
                        if k == (i+1):
                            precisions_at[k] += relevant_count/k
                            recalls_at[k] += relevant_count/total_relevant if total_relevant != 0 else 0
                            map_at[k] += running_percision/k
                        
                            if first_relevant_rank <= k:
                                mrr_at[k] += (1/first_relevant_rank)
                              
                ### QUESTION:   what should be the recall value if there are no relevant documents


            relevant_count = 0
            document_relevance.clear()
            current_query = query

        document_relevance[id] = True if label > RELEVANCE_THRESHOLD else False
    
    # Finish computing metrics
    for k in METRICS_AT_K:
        precisions_at[k] /= queries_count
        recalls_at[k] /= queries_count
        mrr_at[k] /= queries_count
        map_at[k] /= queries_count


# Print metrics
out_file.write("-----------------\n")
out_file.write("RESULTS\n")
out_file.write("File: " + query_file + "\n")
out_file.write("Date: " + str(datetime.now()) + "\n")
out_file.write("@K\t\t\t")
for k in METRICS_AT_K:
    out_file.write(str(k) + "\t")
out_file.write("\n")

statistics = {  
    "PRECISION"   : precisions_at,
    "RECALL   "   : recalls_at,
    "MRR      "   : mrr_at,
    "MAP      "   : map_at
}
for name, statistic in statistics.items():
    out_file.write(name + "\t\t")
    for value in statistic.values():
        out_file.write(str(round(value,2)) + "\t")
    out_file.write("\n")

if out_file != sys.stdout:
    out_file.close()


