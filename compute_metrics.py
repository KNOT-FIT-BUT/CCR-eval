#! /usr/bin/env python3

from datetime import datetime
from time import time
from tqdm import tqdm
import argparse
import math
import sys
import os

from args_compute_metrics import parser
from config import *

from utils.lemmatize import ModelLoadError, ModelNotLoadedError, TokenizerError
from utils.lemmatize import Lemmatizer
from utils.stats import IndexStats
from utils.search import IndexSearcher
from utils.formatting import print_stats, print_grid_table, plot_grid_search, print_stats_csv, print_grid_csv


if __name__ == "__main__":
    args = parser.parse_args()
    stats_format = args.format

    K1_values = [K1_default]
    B_values = [B_default]

    if args.grid_search:
        K1_values = K1_grid_values
        B_values = B_grid_values

    load_morho_model = False
    if args.lemmatize_query:
        load_morho_model = True

    index_stats = IndexStats(args.index_type, load_morpho_model=load_morho_model)

    stats = {}
    total_runs = len(K1_values) * len(B_values)
    
    current_run = 1
    for k1 in K1_values:
        for b in B_values:
            stats[(k1,b)], queries_count = index_stats.calculate_stats(
                args.index_path,
                args.query_file, 
                args.lemmatize_query, 
                k1=k1, b=b,
                current_run=current_run,
                total_runs=total_runs,
                id_url_pairs=args.pairs
            )

            current_run += 1 

    # Ability to write both to stdout and output file
    out_file = open(args.out_path, "w") if args.out_path else sys.stdout
    
    if args.grid_search:
        # TODO Export to csv, plot graphs
        print_grid_table(stats, METRICS_AT_K, queries_count, args.query_file, args.index_path, out_file)
        print_grid_csv(stats, METRICS_AT_K, out_file)
        for k in METRICS_AT_K:
            plot_name = f"dev_gs_precison@{k}.png"
            plot_grid_search(stats, plot_name, k=k)
            logger.info("Saved plot to " + plot_name)
        
    else:
        if stats_format == "table":
            print_stats(stats, METRICS_AT_K, queries_count, args.query_file, args.index_path, out_file)
        elif stats_format == "graph":
            # TODO
            pass
        elif stats_format == "csv":
            print_stats_csv(stats, METRICS_AT_K, queries_count, )

if out_file != sys.stdout:
    out_file.close()