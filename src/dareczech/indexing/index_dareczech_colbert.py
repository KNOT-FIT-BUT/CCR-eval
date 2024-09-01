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

parser.add_argument(
    "--sparse-reduce-type",
    required=False,
    type=str,
    action="store",
    dest="sparse_reduce_type",
    choices=["threshold", "prob_cutoff", "top_k"],
    default="threshold"
)

parser.add_argument(
    "--sparse-reduce-delta",
    required=False,
    type=float,
    action="store",
    dest="sparse_reduce_delta",
    default=0.2
)

parser.add_argument(
    "--sparse-reduce-quantile",
    required=False,
    type=float,
    action="store",
    dest="sparse_reduce_quantile",
    default=0.75
)

parser.add_argument(
    "--sparse-reduce-k",
    required=False,
    type=int,
    action="store",
    dest="sparse_reduce_k",
    default=50
)

parser.add_argument(
    "--sparse-reduce-grid-search", 
    required=False,
    action="store",
    nargs="+",
    dest="sparse_reduce_grid_search"
)

parser.add_argument(
    "--lmbd", 
    required=True, 
    dest="lmbd",
    action="store",
    type=float
)

args = parser.parse_args()

checkpoint = args.checkpoint
collection_path = args.collection
index_name = args.index_name


nbits = 2   # encode each dimension with 2 bits

def main(delta=None, quantile=None, k=None, index_name:str=None):
    print("Loading collection..")
    collection = Collection(path=collection_path)
    print("Collection loaded.")
    print("Passsages", len(collection))

    with Run().context(RunConfig(nranks=1)):  
        config = ColBERTConfig(nbits=nbits,
            bsize=512,
            kmeans_niters=6,
            doc_maxlen=args.max_doclen,
            overwrite=True,
            sparse_reduce=True if args.sparse_reduce_type else False,
            sparse_reduce_type=args.sparse_reduce_type,
            sparse_reduce_delta=delta if delta else args.sparse_reduce_delta,
            sparse_reduce_quantile=quantile if quantile else args.sparse_reduce_quantile,
            sparse_reduce_k=k if k else args.sparse_reduce_k
        )

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(
                        name=args.index_name if index_name is None else index_name,
                        collection=collection
                )

if __name__ == "__main__":
    mp.freeze_support()
    
    if args.sparse_reduce_grid_search:
        print("Starting grid search..")
        preformat_index_name = "exp_{}-{}_sigmoid-lmbd-{}_dareczech_100k_en_colbertv2_sparse.{}.0/"
        for val in args.sparse_reduce_grid_search:
            val = float(val)
            print("Indexing with value:", val)
            print("Index name:", preformat_index_name.format(args.sparse_reduce_type, val, args.lmbd, args.max_doclen))
            main(
                    delta=val, 
                    quantile=val,
                    k=int(val), 
                    index_name=preformat_index_name.format(args.sparse_reduce_type, val, args.lmbd, args.max_doclen)
            )
            
    else:
        main()
