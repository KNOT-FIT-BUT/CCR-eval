from colbert.indexing.codecs.residual import ResidualCodec
from colbert.infra import ColBERTConfig
from utils.collection import load_pairs
from functools import partialmethod
from colbert import Searcher
from tqdm import tqdm
from time import time


class ColBERTIndexSearcher:
    pairs = None 
    def __init__(self, index_path:str, id_url_pairs:str=None):
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        self.searcher = Searcher(index=index_path)
        if id_url_pairs:
            self.pairs = load_pairs(id_url_pairs)

    def _retrieve_results(self, results, k:int, include_content:bool=True, unique_ids:bool=False) -> list:
        results_out = []
        retrieved_ids = set()

        for passage_id, _, _ in zip(*results):
            doc_content = self.searcher.collection[passage_id] if include_content else None
            passage_id = self.pairs.get(int(passage_id)).split("|")[0] if self.pairs else passage_id.split("|")[0]
            # Skip duplicates
            if unique_ids and passage_id in retrieved_ids:
                continue
            results_out.append((passage_id, doc_content))
            retrieved_ids.add(passage_id)
            if len(results_out) == k:
                break
        return results_out

    def search(self, query:str, k:int = 10, include_content:bool=True, unique_ids:bool=False):
            start_time = time()
            results = self.searcher.search(query, k=k)
            self.last_search_time = time() - start_time

            retry = 1
            while True:
                results_out = self._retrieve_results(results, k, include_content, unique_ids)

                if len(results_out) != k:
                    results = self.searcher.search(query, k=k*(retry+1))
                    retry += 1
                else:
                    break
                
            return results_out

    def get_last_search_time(self):
        return self.last_search_time