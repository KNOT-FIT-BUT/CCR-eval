import torch
from transformers import AutoModel, AutoTokenizer
from time import time
from utils.collection import load_pairs
import os

class RawTensorsSearcher:
    pairs = None
    def __init__(self, index_path:str, model_name:str="Seznam/dist-mpnet-czeng-cs-en", dim:int=256, id_url_pairs:str=None, **kwargs):
        if "simcse" in index_path:
            model_name = "Seznam/dist-mpnet-czeng-cs-en"
        elif "contriever" in index_path:
            model_name = "facebook/contriever-msmarco"
        
        self.embeddins_path = index_path
        self.last_search_time = 0
        self.max_length = kwargs.get("max_length") if kwargs.get("max_length") else 512
        self.use_offsets = True if kwargs.get("use_offsets") == True else False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__load_model(model_name)
        self.__load_index()
        self.collection = {}

        if "collection_path" in kwargs:
            self.__load_collection(kwargs["collection_path"])

        if id_url_pairs:
            print(f"Loading pairs ({id_url_pairs})")
            self.pairs = load_pairs(id_url_pairs)
    

    def __load_model(self, model_name:str):
        print("Loading model:", model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def __load_index(self):
        try:
            self.embeddings = torch.load(self.embeddins_path)
            self.embeddings = self.embeddings.to(self.device)
            if self.use_offsets:
                doclens = torch.load(os.path.join(os.path.dirname(self.embeddins_path), "docs_sentence_lens.pt"))
                self.offsets = torch.cumsum(doclens, 0)
        except FileNotFoundError:
            raise FileNotFoundError(f"Embeddings file not found in {self.embeddins_path}")
    
    def __load_collection(self, path):
        print("Loading collection")
        self.collection = {}
        with open(path, "r") as f:
            for line in f:
                try:
                    id, line = line.strip().split("\t")
                except:
                    line = ""
                self.collection[int(id)] = line
    
    def __idx_to_docid(self, idx:int):
        return torch.searchsorted(self.offsets, idx, right=True).item()

    def search(self, query:SyntaxError, k:int=10, include_content:bool=False, **kwargs):
        query = self.tokenizer(query, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt').to(self.device)
        query = self.model(**query).last_hidden_state[:, 0].to(self.device)
        
        start_time = time()
        similarities =  torch.nn.functional.cosine_similarity(self.embeddings, query, dim=1).to(self.device)
        self.last_search_time = time() - start_time
        
        doc_indeces = torch.argsort(similarities, descending=True).tolist()                         
        retrievedNum = 0
        i = 0
        doc_ids = []
        while(retrievedNum < k):
            doc_id = self.__idx_to_docid(doc_indeces[i]) if self.use_offsets else doc_indeces[i]
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
                retrievedNum += 1
            i += 1
            
        results_out = []
        for id in doc_ids:
            id = self.pairs.get(int(id)).split("|")[0] if self.pairs else id
            if include_content:
                results_out.append((id, self.collection.get(id)))
            else:
                results_out.append((id, None))
        return results_out

    def get_last_search_time(self):
        return self.last_search_time


# test = RawTensorsSearcher(index_path="/mnt/minerva1/nlp/projects/CCR/indexes/dareczech_100k_simcse-dist-mpnet-paracrawl/all_cls_embeddings.pt",
#                           model_name="Seznam/simcse-dist-mpnet-paracrawl-cs-en", id_url_pairs="/mnt/minerva1/nlp/nlp/projects/CCR/dareczech/documents/sliced/test/100k/dareczech_100k.pairs.tsv")

# results = test.search("auto", k=10)
# print(results)
# print("Done")
