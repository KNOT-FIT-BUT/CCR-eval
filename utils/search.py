from abc import ABC, abstractmethod

from config import INDEX_TYPES

class IndexSearcher:
    def __new__(cls, index_path: str, index_type: str, **kwargs):
        if index_type not in INDEX_TYPES:
            raise NotImplementedError("Selected index type is not supported")
        if index_type == "bm25":
            from utils.searchers.search_BM25 import BM25IndexSearcher
            return BM25IndexSearcher(index_path, **kwargs)
        elif index_type == "colbert":
            from utils.searchers.search_colbert import ColBERTIndexSearcher
            return ColBERTIndexSearcher(index_path, **kwargs)
        elif index_type == "splade":
            from utils.searchers.search_splade import SpladeIndexSearcher
            return SpladeIndexSearcher(index_path, **kwargs)
        elif index_type == "openaiada":
            from utils.searchers.search_ada import OpenAIADAIndexSearcher
            return OpenAIADAIndexSearcher(index_path, **kwargs)
    
    @abstractmethod
    def search(self, query: str, k: int = 10, include_content=True, unique_ids:bool=False):
        pass

    @abstractmethod
    def get_last_search_time():
        pass

    # Only BM25
    @abstractmethod
    def adjust_bm25_params(self, k1:float, b:float):
        pass
