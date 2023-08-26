from pyserini.search.lucene import LuceneSearcher
from colbert.infra import ColBERTConfig
from colbert import Searcher
from colbert.indexing.codecs.residual import ResidualCodec

from functools import partialmethod
from tqdm import tqdm
from time import time
import json

from config import INDEX_TYPES, logger
from utils.collection import load_collection

class IncorrectIndexType(Exception):
    pass

class IndexSearcher():

    last_search_time = -1 

    def __init__(self, index:str, index_type:str, collection=""):
        if index_type not in INDEX_TYPES:
            raise IncorrectIndexType
        
        if index_type == "bm25":
            index = LuceneSearcher(index)
            self.search_index = index.search 

        elif index_type == "colbert":
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
            logger.info("[COLBERT] Loading doc collection..")
            self.doc_subset, self.collection = load_collection(collection)
            searcher = Searcher(index=index, collection=self.collection)
            logger.info("[COLBERT] Loaded.") 

            self.search_index = searcher.search_all

        self.index_type = index_type

    def search(self, query:str, k:int=10, include_content=True):
        results_out = []

        query = query.lower()

        if self.index_type == "bm25":
            start_time = time()
            results = self.search_index(query, k=k)
            self.last_search_time = time() - start_time
            
            for doc in results:                
                doc_id = doc.docid
                doc_content = json.loads(str(doc.raw))['contents'] if include_content else ""
                results_out.append((doc_id, doc_content))

        elif self.index_type == "colbert":
            query_id = f"{id}_{k}"
            
            start_time = time()
            raw_scores = self.search_index({query_id:query}, k=k)
            self.last_search_time = time() - start_time

            for didx, rank, score in raw_scores.data[query_id]:
                doc_content = self.doc_subset[didx]["text"] if include_content else ""
                results_out.append((self.doc_subset[didx]["id"], doc_content))
             
        return results_out
    

    def get_last_search_time(self):
        return self.last_search_time
