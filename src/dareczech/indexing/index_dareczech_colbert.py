#! /usr/bin/env python


# sys.path.insert(0, '../')

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
        "--checkpoint", 
        required=True, 
        type=str,
        dest="checkpoint"
)

parser.add_argument(
        "--collection", 
        required=True, 
        type=str,
        dest="collection"
)

parser.add_argument(
        "--index-name", 
        required=True, 
        type=str, 
        dest="index_name"
)

parser.add_argument(
        "--max-doclen", 
        required=True,
        type=int,
        dest="max_doclen"
)

args = parser.parse_args()

checkpoint = args.checkpoint
collection_path = args.collection
index_name = args.index_name

nbits = 2   # encode each dimension with 2 bits

def main():
    print("Loading collection..")
    collection = Collection(path=collection_path)
    print("Collection loaded.")
    print("Passsages", len(collection))

    with Run().context(RunConfig(nranks=1)):  
        config = ColBERTConfig(nbits=nbits,
                bsize=512,
                #kmeans_niters=6,
                doc_maxlen=args.max_doclen,
                overwrite=True)

        indexer = Indexer(checkpoint=checkpoint, config=config)
        #indexer.prepare(name=index_name, collection=collection, overwrite=True)
        indexer.index(name=index_name, collection=collection)

if __name__ == "__main__":
    mp.freeze_support()
    main()
