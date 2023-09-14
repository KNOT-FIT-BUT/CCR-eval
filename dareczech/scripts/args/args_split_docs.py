import argparse 

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--doc-file", 
    required=True,
    action="store",
    help="Input doc file",
    dest="input_docs"
)

parser.add_argument(
    "-o", "--output", 
    required=True, 
    action="store", 
    help="Dest split-docs file",
    dest="output_docs"
)

parser.add_argument(
    "--tokenizer", 
    required=True,
    action="store", 
    help="HF tokenizer",
    dest="tokenizer"
)

parser.add_argument(
    "-t", "--token-threshold", 
    required=True,
    action="store", 
    help="Split threshold in tokens (split above this number)",
    dest="split_threshold"
)

parser.add_argument(
    "--overlap", 
    required=True,
    action="store", 
    help="Doc split overlap",
    dest="overlap"
)

parser.add_argument(
    "--doc-id-key", 
    required=False,
    action="store", 
    default="url",
    help="Dictionary (json) key to document id (default 'id')",
    dest="doc_id_key"
)
