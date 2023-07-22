import os
import re

from lemmatize import Lemmatizer
from dareczech_reg import DOC_TITLE_REG, DOC_URL_REG, DOC_BTE_REG

INPUT_PATH = "dareczech/dev.tsv"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError

lemmatizer = Lemmatizer()
lemmatizer.load_model()

with open(INPUT_PATH) as file_in:
    # Skip first line (tsv header)
    next(file_in)

    for line in file_in:
        data = line.split("\t")
        
        # id      = data[0]
        # query   = data[1]
        # url     = data[2]
        doc     = data[3]
        # title   = data[4]
        # label   = data[5]

        # doc_title   = re.search(DOC_TITLE_REG, doc).group(1)
        # doc_url     = re.search(DOC_URL_REG, doc).group(1)
        doc_bte     = re.search(DOC_BTE_REG, doc).group(1)
        lemmatized = lemmatizer.lemmatize_text(doc_bte)
        print(lemmatized)
        break

lemmatizer = Lemmatizer()

