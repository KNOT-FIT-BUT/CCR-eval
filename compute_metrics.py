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
from utils.stats import IndexStats
from utils.search import IndexSearcher
from utils.formatting import print_stats, print_grid_table, plot_grid_search

from args_compute_metrics import parser
from config import *

if __name__ == "__main__":
    args = parser.parse_args()

    K1_values = [K1_default]
    B_values = [B_default]

# Perform i/o file checks
if not os.path.exists(query_file):
    logger.error("Query path does not exist, stopping..")
    exit(1)
elif not query_file.endswith(".tsv"):
    print(query_file)
    logger.warning("Query file might not be in the correct format")

    index_stats = IndexStats(args.index_type, load_morpho_model=True)

    stats = {}
    total_runs = len(K1_values) * len(B_values)
    
    current_run = 1
    for k1 in K1_values:
        for b in B_values:
            stats[(k1,b)], queries_count = index_stats.calculate_stats(
                args.index_path,
                args.query_file, 
                args.collection, 
                args.lemmatize_query, 
                k1=k1, b=b,
                current_run=current_run,
                total_runs=total_runs
            )

            current_run += 1 

    # Ability to write both to stdout and output file
    out_file = open(args.out_path, "w") if args.out_path else sys.stdout
    
    if args.grid_search:
        # TODO Export to csv, plot graphs
        print_grid_table(stats, METRICS_AT_K, queries_count, args.query_file, args.index_path, out_file)
        logger.info("Saved grid table to: " + args.out_path)
        for k in METRICS_AT_K:
            plot_name = f"dev_gs_precison@{k}.png"
            plot_grid_search(stats, plot_name, k=k)
            logger.info("Saved plot to " + plot_name)
        
    else:
        print_stats(stats, METRICS_AT_K, queries_count, args.query_file, args.index_path, out_file)

if out_file != sys.stdout:
    out_file.close()