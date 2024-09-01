from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer

import os

MS_MARCO_PATH = "/mnt/data/xsteti05_ccr/CCR/datasets/msmarco"

if not os.path.exists(MS_MARCO_PATH):
    MS_MARCO_PATH = "/mnt/minerva1/nlp/projects/CCR/datasets/msmarco"
    
MS_MARCO_PATH = "/mnt/minerva1/nlp-2/homes/xsteti05/mnt/karolina/msmarco"

def train():
    # use 4 gpus (e.g. four A100s, but you can use fewer by changing nway,accumsteps,bsize).
    with Run().context(RunConfig(nranks=2)):
        triples = os.path.join(MS_MARCO_PATH, 'examples.colbert.json')
        queries = os.path.join(MS_MARCO_PATH, 'queries.train.tsv')
        collection = os.path.join(MS_MARCO_PATH, 'collection.tsv')
        
        triples = os.path.join(MS_MARCO_PATH, 'sliced/examples.64.1k.json')
        queries = os.path.join(MS_MARCO_PATH, 'sliced/queries.train.10k.tsv')
        collection = os.path.join(MS_MARCO_PATH, 'sliced/collection.10k.tsv')   


        config = ColBERTConfig(bsize=32, lr=1e-05, warmup=20_000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False, nway=64, 
                               accumsteps=1, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        trainer.train(checkpoint='colbert-ir/colbertv1.9')  # or start from scratch, like `bert-base-uncased`


if __name__ == '__main__':
    train()
