from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer

import argparse
import os

MSMARCO_PATH = "/home/xsteti05/mnt/karolina/msmarco/"

parser = argparse.ArgumentParser()
parser.add_argument(
        "--checkpoint", 
        required=True,
        action="store", 
        dest="checkpoint"
        )

def train(checkpoint:str):
    with Run().context(RunConfig(nranks=4)):
        triples = os.path.join(MSMARCO_PATH, 'examples.64.json') 
        queries = os.path.join(MSMARCO_PATH, 'queries.train.tsv')
        collection = os.path.join(MSMARCO_PATH, 'collection.tsv')

        #triples = os.path.join(MSMARCO_PATH, 'sliced/examples.64.1k.json')  
        #queries = os.path.join(MSMARCO_PATH, 'sliced/queries.train.10k.tsv')  
        #collection = os.path.join(MSMARCO_PATH, 'sliced/collection.10k.tsv')


        config = ColBERTConfig(
                bsize=32, lr=1e-05, warmup=20_000,
                doc_maxlen=180, dim=128, attend_to_mask_tokens=False, 
                nway=64, accumsteps=1, similarity='cosine', use_ib_negatives=True,
                lmbd=1.0 
                )
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        trainer.train(checkpoint=checkpoint)

if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if not os.path.exists(checkpoint):
        print("Checkpoint path does not exist")
        exit(1)

    train(checkpoint)
