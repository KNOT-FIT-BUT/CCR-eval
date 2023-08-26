#! /usr/bin/env python3

from datetime import datetime
from time import time
from tqdm import tqdm
import argparse
import logging
import math
import json
import sys
import os

from pyserini.search.lucene import LuceneSearcher
from utils.lemmatize import ModelLoadError, ModelNotLoadedError, TokenizerError
from utils.lemmatize import Lemmatizer

from config import *

def format_table_line(input_line:list, n=10, delim="|", out_stream=sys.stdout):
    for i, item in enumerate(input_line):
        out_stream.write(f"{item:^{n}}")
        if i+1 != len(input_line):
            out_stream.write(f" {delim} ")
    out_stream.write("\n")

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Anything with score above 0 considered relevant
RELEVANCE_THRESHOLD = 0.0 

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--query-file",
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
elif not query_file.endswith(".tsv"):
    print(query_file)
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
                        
logging.info("Preparig file..")
lines_count = sum(1 for line in open(query_file)) - 1
with open(query_file) as qrel_file:

    # Skip query file header
    next(qrel_file)

    current_query = ""
    relevant_docs = {}
    queries_count = 0

    # Metrics
    relevant_count = 0

    precisions_at = {}
    recalls_at = {}
    mrr_at = {}
    map_at = {}
    ndcg_at = {}
    exec_time_at = {}

    # Initialize dictionaries
    for k in METRICS_AT_K:
        precisions_at[k] = 0.0
        recalls_at[k] = 0.0
        mrr_at[k] = 0.0
        map_at[k] = 0.0
        ndcg_at[k] = 0.0
        exec_time_at[k] = 0.0

    for line in tqdm(qrel_file, total=lines_count, desc="Computing", unit="queries"):
        data = line.split("\t")

        id, query, url, label = data
        label = float(label)

        # New test-query
        if query != current_query:

            # Perform current query search
            if current_query:
                queries_count += 1

                current_query = lemmatizer.lemmatize_text(current_query)

                # Perform searches for top-k
                for k in METRICS_AT_K:
                    start_time = time()
                    results = searcher.search(current_query, k=k)
                    exec_time_at[k] += time() - start_time
                
                    total_relevant = len(relevant_docs)
                    sorted_scores = sorted(relevant_docs.values(), reverse=True)

                    relevant_count = 0
                    running_percision = 0
                    first_relevant_rank = 0
                    running_dcg = 0
                    running_idcg = 0

                    # Go through results
                    for i, result in enumerate(results):
                        result_id = result.docid

                        # Retrieved document is relevant
                        if result_id in relevant_docs.keys():
                            relevant_count += 1

                            # If first relevant document
                            if relevant_count == 1:
                                first_relevant_rank = (i+1)
                        
                            # DCG only counted when relevant, else is 0
                            relevancy = float(relevant_docs.get(result_id))
                            running_dcg += relevancy/math.log2((i+1) + 1)

                        running_idcg += sorted_scores[i]/math.log2((i+1) + 1) if i < len(sorted_scores) else 0
                        running_percision += relevant_count/(i+1)

                    precisions_at[k] += relevant_count/k
                    recalls_at[k] += relevant_count/total_relevant if total_relevant != 0 else 0
                    map_at[k] += running_percision/k
                    ndcg_at[k] += running_dcg/running_idcg if running_idcg != 0 else 0
                
                    if first_relevant_rank <= k:
                        mrr_at[k] += (1/first_relevant_rank) if first_relevant_rank != 0 else 0

            relevant_docs.clear()
            current_query = query
        
        if label > RELEVANCE_THRESHOLD:
            relevant_docs[url] = label
    
    # Finish computing metrics
    for k in METRICS_AT_K:
        precisions_at[k] /= queries_count
        recalls_at[k] /= queries_count
        mrr_at[k] /= queries_count
        map_at[k] /= queries_count
        exec_time_at[k] /= queries_count
        ndcg_at[k] /= queries_count



# Print metrics
out_file.write("-------------------------\n")
out_file.write("File:  " + query_file + "\n")
out_file.write("Index: " + index_file + "\n")
out_file.write("Date:  " + str(datetime.now()) + "\n\n")

format_table_line(["@K"] + [str(val) for val in METRICS_AT_K], n=15, out_stream=out_file)
out_file.write(7*15*"_" + "\n")

statistics = {  
    "PRECISION"         : precisions_at,
    "RECALL"            : recalls_at,
    "MRR"               : mrr_at,
    "MAP"               : map_at,
    "NDCG"              : ndcg_at,
    "EXEC TIME [ms]"    : exec_time_at
}

for name, statistic in statistics.items():
    line_values = [name]
    for value in statistic.values():        
        
        # Convert exec_time to milliseconds
        if statistic == exec_time_at:            
            value = value * 10**3

        line_values.append('{:.2f}'.format(round(value, 2)))

    format_table_line(line_values, n=15, out_stream=out_file)
    out_file

if out_file != sys.stdout:
    out_file.close()


