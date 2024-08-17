from openai.embeddings_utils import get_embedding, cosine_similarity
from time import time
import pandas as pd
import json
import os

class MissingOpenAIAPIkey(Exception):
    pass

class OpenAIADAIndexSearcher:
    def __init__(self, index_path:str, api_key=None, embedding_model="text-embedding-ada-002", id_url_pairs:str=None):
        self.df = pd.read_pickle(os.path.join(index_path, os.path.basename(index_path)+".pickle"))
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")  
        self.embedding_model = embedding_model

        if self.api_key == None:
            raise MissingOpenAIAPIkey
        


    def search(self, query: str, k: int = 10, include_content=True, **kwargs):
        results_out = []
        
        if query.startswith("[") and query.endswith("]"):
            embedding = json.loads(query)
        else:
            embedding = get_embedding(query, engine=self.embedding_model, api_key=self.api_key)

        start_time = time()
        self.df['similarities'] = self.df.embedding.apply(lambda x: cosine_similarity(x, embedding))
        self.last_search_time = time() - start_time
        
        for _, row in self.df.sort_values("similarities", ascending=False).head(k).iterrows():
            row_content = (row["url"], row["doc"]) if include_content else (row["url"], None)
            results_out.append(row_content)
        return results_out

    # def search(self, query: str, k: int = 10, include_content=True, **kwargs):
    #     results_out = []
    #     embedding = get_embedding(query, engine=self.embedding_model, api_key=self.api_key)
        
    #     start_time = time()
        
    #     df_gpu = cudf.DataFrame(self.df)
        
    #     similarities_gpu = cuml.metrics.pairwise.cosine_similarity(df_gpu['embedding'].tolist(), [embedding])
    #     similarities = similarities_gpu.to_array().squeeze()
        
    #     self.df['similarities'] = similarities
    #     self.last_search_time = time() - start_time

    #     top_k_rows = self.df.nlargest(k, 'similarities')
        
    #     for _, row in top_k_rows.iterrows():
    #         row_content = (row["url"], row["doc"]) if include_content else (row["url"], None)
    #         results_out.append(row_content)
        
    #     return results_out


    
    def get_last_search_time(self):
        return self.last_search_time
