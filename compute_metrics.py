#! /usr/bin/env python3

from datetime import datetime
from time import time
from tqdm import tqdm
import argparse
import math
import sys
import os

from utils.lemmatize import ModelLoadError, ModelNotLoadedError, TokenizerError
from utils.lemmatize import Lemmatizer
from utils.search import IndexSearcher
from utils.formatting import format_table_line

from args_compute_metrics import parser
from config import *

args = parser.parse_args()

query_file = args.query_file
index_file = args.index_path
out_path = args.out_path
index_type = args.index_type
collection = args.collection
lemmatize_query = args.lemmatize_query

# Perform i/o file checks
if not os.path.exists(query_file):
    logger.error("Query path does not exist, stopping..")
    exit(1)
elif not query_file.endswith(".tsv"):
    print(query_file)
    logger.warning("Query file might not be in the correct format")


# Ability to write both to stdout and output file
out_file = open(out_path, "w") if out_path else sys.stdout

logger.info("INDEX TYPE: " + index_type)
   
searcher = IndexSearcher(index_file, index_type, collection=collection)

if lemmatize_query:
    logger.info("Loading morpho model...")
    try:
        lemmatizer = Lemmatizer(MORPHODITA_MODEL)
        lemmatizer.load_model()
    except FileNotFoundError:
        logger.error("Morpho model not found.")
        exit(1)
    except ModelLoadError:
        logger.error("Error while loading czech morpho model.")
        exit(1)
    logger.info("Model loaded.")
                        
logger.info("Preparig file..")
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

                if lemmatize_query:
                    current_query = lemmatizer.lemmatize_text(current_query)

                # Perform searches for top-k
                for k in METRICS_AT_K:
                    
                    results = searcher.search(query=current_query, k=k, include_content=False)
                    exec_time_at[k] += searcher.get_last_search_time()
                
                    total_relevant = len(relevant_docs)
                    sorted_scores = sorted(relevant_docs.values(), reverse=True)

                    relevant_count = 0
                    running_percision = 0
                    first_relevant_rank = 0
                    running_dcg = 0
                    running_idcg = 0

                    # Go through results
                    for i, result in enumerate(results):
                        result_id = result[0]

                        # Retrieved document is relevant
                        if result_id in relevant_docs.keys():
                            relevant_count += 1

                            # If first relevant document
                            if relevant_count == 1:
                                first_relevant_rank = (i+1)
                        
                            # DCG only counted when relevant, else 0
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
out_file.write("Date: " + str(datetime.now()) + "\n")
out_file.write("Query count: " + str(queries_count) + "\n")
out_file.write("Qrel: " + query_file + "\n")
out_file.write("Index: " + index_file + "\n")

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


