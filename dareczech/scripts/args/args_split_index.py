import argparse 

parser = argparse.ArgumentParser()

parser.add_argument(
    "-qr", "--qrel", 
    required=True,
    action="store",
    help="Qrel file",
    dest="qrel"
)

parser.add_argument(
    "-d", "--doc-file", 
    required=True,
    action="store",
    help="Doc file (converted for pyserini)",
    dest="docs"
)

parser.add_argument(
    "-s", "--index-slice", 
    required=True, 
    action="store", 
    help="Size of the sliced index",
    dest="slice"
)

parser.add_argument(
    "--doc-id-key", 
    required=False,
    action="store", 
    help="Dictionary (json) key to document id (default 'id')",
    dest="doc_id_key"
)

parser.add_argument(
    "-o", "--output", 
    required=True, 
    action="store", 
    help="Dest docs file",
    dest="output"
)
