#! /usr/bin/env python3

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import json
import os
import re

from utils import get_document_stats
from args.args_dareczech_distribution import parser
from config import MAX_CHAR_THRESHOLD, MAX_WORDS_THRESHOLD, MAX_SENTENCE_THRESHOLD

args = parser.parse_args()
input_file = args.doc_file
out_dir = args.out_dir

if not os.path.exists(input_file):
    print("Error: input doc file does not exist...")
    exit(1)


char_distr = []
words_distr = []
sentences_distr = []

doc_counts = [0, 0, 0]

with open(input_file) as doc_file:
    doc_count = 0
    for line in doc_file:
        print("Doc count:", doc_count, end="\r")
        line = json.loads(line)
        
        doc = line['doc']
        chars, words, sentences = get_document_stats(doc)
        

        if words < MAX_CHAR_THRESHOLD:
            char_distr.append(chars) 
            doc_counts[0] += 1
        
        if words < MAX_WORDS_THRESHOLD:
            words_distr.append(words)
            doc_counts[1] += 1
        
        if sentences < MAX_SENTENCE_THRESHOLD:
            sentences_distr.append(sentences)
            doc_counts[2] += 1

        doc_count += 1
    print()

char_distr = np.array(char_distr)
words_distr = np.array(words_distr)
sentences_distr = np.array(sentences_distr)

distributions_data = [char_distr, words_distr, sentences_distr]
distributions_names = ["Characters", "Words", "Sentences"]

# Plot distributions
print("Saving...")
for i, data in enumerate(distributions_data):
    name = distributions_names[i]

    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 100)
    y_vals = kde(x_vals)

    plt.figure(figsize=(10,6))
    plt.plot(x_vals, y_vals * doc_counts[i], color='blue')
    # plt.hist(data, bins=10, edgecolor="black")
    plt.ylabel("Number of documents")
    plt.xlabel(f"Number of {name.lower()}")
    plt.title(f"{name} counts distribution - {os.path.basename(input_file)}")
    
    input_file_name = os.path.splitext(os.path.basename(input_file))[0]

    file_name = f"{input_file_name}_{name.lower()}_distribution.png"
    file_name = f"{out_dir.rstrip('/')}/{file_name}" if out_dir else file_name

    plt.savefig(file_name)
    print("Saved to:", file_name)

print("Done.")
