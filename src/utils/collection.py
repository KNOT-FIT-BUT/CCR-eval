# from colbert.data import Collection
# import json

def load_pairs(id_url_pairs:str) ->dict:
    with open(id_url_pairs) as pairs_file:
        pairs = {}
        for line in pairs_file:
            id, url = line.rstrip().split("\t")
            pairs[int(id)] = url
    return pairs

