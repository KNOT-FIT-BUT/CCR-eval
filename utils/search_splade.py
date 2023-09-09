from utils.splade_funcs import to_list, numba_score_float, select_topk
from splade.datasets.datasets import CollectionDatasetPreLoad
from splade.datasets.dataloaders import CollectionDataLoader
from splade.models.transformer_rep import Splade, SpladeDoc
from splade.indexing.inverted_index import IndexDictOfArray
from transformers import AutoTokenizer
from collections import defaultdict
from time import time
import numpy as np
import pickle
import torch
import numba
import os

class SpladeIndexSearcher:
    def __init__(self, index_path:str, embedding_model="naver/splade-cocondenser-selfdistil"):
        self. model = Splade(model_type_or_dir=embedding_model)
        self.device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.collection_size = len(pickle.load(open(os.path.join(index_path, "doc_ids.pkl"),"rb")))
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=embedding_model)
        self.sparse_index = IndexDictOfArray(self.index_path, dim_voc=self.model.output_dim)
        self.doc_ids = pickle.load(open(os.path.join(index_path, "doc_ids.pkl"), "rb"))
        self.index_doc_ids = numba.typed.Dict()
        self.index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.index_doc_values[key] = value

    def search(self, query: str, k: int = 10, include_content=True):
            with torch.no_grad():
                results_out = []
                query = [query]

                start_time = time()
                processed_passage = self.tokenizer(query,
                                    add_special_tokens=True,
                                    padding="longest", 
                                    truncation="longest_first", 
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
                self.last_search_time = time() - start_time 

                # TODO return with doc_content
                for id_ in filtered_indexes:
                    results_out.append((id_, None))
            return results_out
    
    def get_last_search_time(self):
        return self.last_search_time