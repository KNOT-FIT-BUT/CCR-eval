from colbert.indexing.codecs.residual import ResidualCodec
from colbert.infra import ColBERTConfig
from colbert import Searcher
from functools import partialmethod
from tqdm import tqdm
from time import time


class ColBERTIndexSearcher:
    def __init__(self, index_path:str):
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        self.searcher = Searcher(index=index_path)


    def search(self, query: str, k: int = 10, include_content=True):
            results_out = []

            start_time = time()
            results = self.searcher.search(query, k=k)
            self.last_search_time = time() - start_time
            
            for passage_id, _, _ in zip(*results):
                doc_content = self.searcher.collection[passage_id] if include_content else None
                results_out.append((passage_id, doc_content))
            return results_out

    def get_last_search_time(self):
        return self.last_search_time