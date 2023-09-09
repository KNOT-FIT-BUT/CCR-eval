# BM25
from pyserini.search.lucene import LuceneSearcher

# ColBERT
from colbert.indexing.codecs.residual import ResidualCodec
from colbert.infra import ColBERTConfig
from colbert import Searcher

# SPLADE
from .splade_funcs import to_list, numba_score_float, select_topk
from splade.datasets.datasets import CollectionDatasetPreLoad
from splade.datasets.dataloaders import CollectionDataLoader
from splade.models.transformer_rep import Splade, SpladeDoc
from splade.indexing.inverted_index import IndexDictOfArray
from transformers import AutoTokenizer
from collections import defaultdict
import pickle
import torch
import numba

# OpenAI
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd

from functools import partialmethod
from tqdm import tqdm
from time import time
import numpy as np
import json
import os

from config import INDEX_TYPES, logger
from utils.collection import load_collection

class IndexSearcher():

    # Default values
    last_search_time = -1 
    K1 = 0.9
    B = 0.4

    def __init__(self, index_path:str, index_type:str, collection=""):
        if index_type not in INDEX_TYPES:
            raise NotImplementedError
        
        self.index_type = index_type
        self.index_path = index_path
        self.index = None
        self.collection = collection
        
        if index_type == "bm25":
            self.__init_bm25_searcher()

        elif index_type == "plaidx":
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
            logger.info("[PLAIDX] Loading doc collection..")
            self.doc_subset, self.collection = load_collection(collection)
            searcher = Searcher(index=index_path, collection=self.collection)
            logger.info("[PLAIDX] Loaded.") 

            self.searcher = searcher.search_all

        elif index_type == "colbert":
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
            self.searcher = Searcher(index=index_path)

        elif index_type == "splade":
            self. model = Splade(model_type_or_dir="naver/splade-cocondenser-selfdistil")
            self.device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.collection_size = len(pickle.load(open(os.path.join(index_path, "doc_ids.pkl"),"rb")))
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="naver/splade-cocondenser-selfdistil")
            self.sparse_index = IndexDictOfArray(self.index_path, dim_voc=self.model.output_dim)
            self.doc_ids = pickle.load(open(os.path.join(index_path, "doc_ids.pkl"), "rb"))
            self.index_doc_ids = numba.typed.Dict()
            self.index_doc_values = numba.typed.Dict()
            for key, value in self.sparse_index.index_doc_id.items():
                self.index_doc_ids[key] = value
            for key, value in self.sparse_index.index_doc_value.items():
                self.index_doc_values[key] = value

        elif index_type == "openaiada":
            self.df = pd.read_pickle(index_path)
            self.api_key = os.getenv("OPENAI_API_KEY")

    def __init_bm25_searcher(self):
        self.adjust_bm25_params(self.K1, self.B)

    def adjust_bm25_params(self, k1:float, b:float):
        if self.index_type == "bm25":
            del self.index
            index = LuceneSearcher(self.index_path)
            index.set_bm25(float(k1), float(b))
            self.searcher = index.search 
            self.index = index
        else:
            raise Exception("K1,B params not present for this type of index")

    # Search index
    def search(self, query:str, k:int=10, include_content=True):
        # Results output format - list of tuples
        # [ (docid, doccontent), (docid2, doccontent2), ...]

        results_out = []

        query = query.lower()

        if self.index_type == "bm25":
            start_time = time()
            results = self.searcher(query, k=k)
            self.last_search_time = time() - start_time
            
            for doc in results:                
                doc_id = doc.docid
                doc_content = json.loads(str(doc.raw))['contents'] if include_content else ""
                results_out.append((doc_id, doc_content))

        elif self.index_type == "plaidx":
            query_id = f"{id}_{k}"
            
            start_time = time()
            raw_scores = self.searcher({query_id:query}, k=k)
            self.last_search_time = time() - start_time

            for didx, rank, score in raw_scores.data[query_id]:
                doc_content = self.doc_subset[didx]["text"] if include_content else ""
                results_out.append((self.doc_subset[didx]["id"], doc_content))
        
        elif self.index_type == "colbert":
            start_time = time()
            results = self.searcher.search(query, k=k)
            self.last_search_time = time() - start_time
            
            for passage_id, passage_rank, passage_score in zip(*results):
                doc_content = ""
                if include_content:
                    doc_content = self.searcher.collection[passage_id]
                results_out.append((passage_id, doc_content))
        
        elif self.index_type == "splade":
            with torch.no_grad():
                query = [query]
                processed_passage = self.tokenizer(query,
                                    add_special_tokens=True,
                                    padding="longest",  # pad to max sequence length in batch
                                    truncation="longest_first",  # truncates to self.max_length
                                    max_length=self.tokenizer.model_max_length,
                                    return_attention_mask=True)
                data =  {k: torch.tensor(v) for k, v in processed_passage.items()}
                inputs = {k: v for k, v in data.items()}
                for k,v in inputs.items():
                    inputs[k] = v.to(self.device)
                query = self.model(q_kwargs=inputs)["q_rep"]
                row, col = torch.nonzero(query, as_tuple=True)
                values = query[to_list(row), to_list(col)]
                
                filtered_indexes, scores = numba_score_float(
                                                    self.index_doc_ids,
                                                    self.index_doc_values,
                                                    col.cpu().numpy(),
                                                    values.cpu().numpy().astype(np.float32),
                                                    threshold=0,
                                                    size_collection=self.collection_size
                                                )
                filtered_indexes, scores = select_topk(filtered_indexes, scores, k=5)    

                # TODO return with doc_content
                for id_ in filtered_indexes:
                    results_out.append((id_, None))
                
        elif self.index_type == "openaiada":
            embedding = get_embedding(query, engine='text-embedding-ada-002', api_key=self.api_key)
            self.df['similarities'] = self.df.embedding.apply(lambda x: cosine_similarity(x, embedding))
            for index, row in self.df.sort_values("similarities", ascending=False).head(k).iterrows():
                if include_content:
                    row_content = (row["url"], row["doc"])
                else:
                    row_content = (row["url"], None)
                results_out.append(row_content)
 
        return results_out
        

    def get_last_search_time(self):
        return self.last_search_time

