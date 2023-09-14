from datetime import datetime
from time import time
from tqdm import tqdm
import math
import json
import os

from utils.lemmatize import ModelLoadError, ModelNotLoadedError, TokenizerError
from utils.lemmatize import Lemmatizer
from utils.search import IndexSearcher

from args_compute_metrics import parser
from config import *

class IndexStats():
    def __init__(self, index_type:str, load_morpho_model=True):
        if load_morpho_model:
            self.load_morpho_model()
        
        if index_type not in INDEX_TYPES:
            raise Exception(f"Incorrect index type: {index_type}")
        self.index_type = index_type
        logger.info("INDEX TYPE: " + index_type)

    
    # Perform i/o file checks
    def __perform_checks(self, query_file:str):
        if not os.path.exists(query_file):
            logger.error("Query path does not exist, stopping..")
            raise FileNotFoundError
        
        elif not query_file.endswith(".tsv"):
            print(query_file)
            logger.warning("Query file might not be in the correct format")

    def load_morpho_model(self):
        logger.info("Loading morpho model...")
        try:
            self.lemmatizer = Lemmatizer(MORPHODITA_MODEL)
            self.lemmatizer.load_model()
        except FileNotFoundError:
            logger.error("Morpho model not found.")
            exit(1)
        except ModelLoadError:
            logger.error("Error while loading czech morpho model.")
            exit(1)
        logger.info("Model loaded.")
                                

    def calculate_stats(
            self, 
            index_file:str,
            query_file:str,
            lemmatize_query:str, 
            k1:float, b:float,
            current_run:int,
            total_runs:int,
            id_url_pairs:str=None
        ):

        self.__perform_checks(query_file)

        searcher = IndexSearcher(index_path=index_file, index_type=self.index_type, id_url_pairs=id_url_pairs)
        
        if self.index_type == "bm25":
            searcher.adjust_bm25_params(k1=k1, b=b)
    
        lines_count = sum(1 for line in open(query_file)) - 1
        with open(query_file) as qrel_file:   
            top_k_data = {key: {} for key in METRICS_AT_K}
            # Skip query file header
            next(qrel_file)

            current_query = ""
            relevant_docs = {}
            queries_count = 0

            # Metrics
            relevant_count = 0


            # Initialize stats dictionary
            stats_key = (k1,b)
            stats = {
                stats_key:{
                    "precision":{},
                    "recall": {}, 
                    "mrr": {}, 
                    "map": {}, 
                    "ndcg": {}, 
                    "exec_time": {}
                }
            }
            for stat_name in stats[stats_key].keys():
                stats[stats_key][stat_name] = {}
                for k in METRICS_AT_K:
                    stats[stats_key][stat_name][k] = 0.0
                            
            progress_bar_desc = f"Computing [{current_run}/{total_runs}]"

            for line in tqdm(qrel_file, total=lines_count, desc=progress_bar_desc, unit="queries", disable=False):
                data = line.split("\t")

                id, query, url, label = data
                label = float(label)

                # New test-query
                if query != current_query:

                    # Perform current query search
                    if current_query:
                        queries_count += 1

                        if lemmatize_query:
                            current_query = self.lemmatizer.lemmatize_text(current_query)

                        # Perform searches for top-k
                        for k in METRICS_AT_K:
                            
                            results = searcher.search(query=current_query, k=k, include_content=False)
                            stats[stats_key]["exec_time"][k] += searcher.get_last_search_time()

                            top_k_data[k][current_query] = [url for url, doc in results] 
                        
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
                        
                            stats[stats_key]["precision"][k] += relevant_count/k
                            stats[stats_key]["recall"][k] += relevant_count/total_relevant if total_relevant != 0 else 0
                            stats[stats_key]["map"][k] += running_percision/k
                            stats[stats_key]["ndcg"][k] += running_dcg/running_idcg if running_idcg != 0 else 0

                            if first_relevant_rank <= k:
                                stats[stats_key]["mrr"][k] += (1/first_relevant_rank) if first_relevant_rank != 0 else 0

                    relevant_docs.clear()
                    current_query = query
                
                if label > RELEVANCE_THRESHOLD:
                    relevant_docs[url] = label
            
            # Average out stats
            for k in METRICS_AT_K:
                stats[stats_key]["precision"][k] /= queries_count
                stats[stats_key]["recall"][k] /= queries_count
                stats[stats_key]["mrr"][k] /= queries_count
                stats[stats_key]["map"][k] /= queries_count
                stats[stats_key]["ndcg"][k] /= queries_count
                stats[stats_key]["exec_time"][k] /= queries_count
            
            with open(f"{self.index_type}.topk", "w") as topk_file:
                topk_file.write(json.dumps(top_k_data, ensure_ascii=False))
            return stats[stats_key], queries_count
