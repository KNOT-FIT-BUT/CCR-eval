from pyserini.search.lucene import LuceneSearcher
from time import time
import json

from utils.collection import load_pairs

class BM25IndexSearcher():
    K1 = 2.0
    B = 1.0
    def __init__(self, index_path:str, **kwargs):
        self.index_path = index_path
        self.index = None
        self.__init_bm25_searcher()
    
    def __init_bm25_searcher(self):
        self.adjust_bm25_params(self.K1, self.B)

    def adjust_bm25_params(self, k1:float, b:float):
        del self.index
        index = LuceneSearcher(self.index_path)
        index.set_bm25(float(k1), float(b))
        self.searcher = index.search 
        self.index = index

    def search(self, query: str, k: int = 10, include_content:bool=True, **kwargs):
        results_out = []
        start_time = time()
        results = self.searcher(query.lower(), k=k)
        self.last_search_time = time() - start_time
            
        for doc in results:                
            doc_id = doc.docid
            doc_content = json.loads(str(doc.raw))['contents'] if include_content else None
            results_out.append((doc_id, doc_content))
        return results_out

    def get_last_search_time(self):
        return self.last_search_time




