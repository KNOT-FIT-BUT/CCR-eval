from colbert.data import Collection
import json

def load_collection(collection_path:str):
    doc_subset = []
    with open(collection_path) as file:
        for line in file:
            line = json.loads(line)

            doc_dict = {
                "id": line['url'],
                "cc_file": "cc_file",
                "time": None,
                "title": None,
                "text": line['doc'],
                "url": line['url']
            }

            doc_subset.append(doc_dict)

    collection = Collection.cast(doc_subset)
    return doc_subset, collection

