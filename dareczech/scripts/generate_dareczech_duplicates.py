#! /usr/bin/env python3

import argparse
import json
import os

parser = argparse.ArgumentParser(description="Find duplicate URLs in a TSV file and save them to a JSON file.")
parser.add_argument("-d", "--input-dir", required=True, action="store", dest="input_dir", type=str, help="Path to the input TSV file")
parser.add_argument("-o", "--output",required=True, action="store", dest="output_file", type=str, help="Path to the output JSON file")

args = parser.parse_args()

tsv_files = []

if os.path.isdir(args.input_dir):
    for item in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, item)

        if os.path.isfile(item_path) and item.strip().endswith('.tsv'):
            tsv_files.append(item_path)

duplicate_urls = {}

print("Running...")

for tsv_file_path in tsv_files:
    print("Processing", tsv_file_path)
    with open(tsv_file_path, 'r') as tsv_file:
        next(tsv_file)

        for line in tsv_file:
            fields = line.strip().split('\t')
            id, query, url, doc, title, label = fields

            if url in duplicate_urls.keys():
                duplicate_urls[url].append(id)
            else:
                duplicate_urls[url] = [id]

# remove non-duplicate url-id pairs
for url in duplicate_urls.copy():
    if len(duplicate_urls[url]) == 1:
        duplicate_urls.pop(url)

with open(args.output_file, 'w') as json_file:
    json.dump(duplicate_urls, json_file, ensure_ascii=False)

print("Duplicate URLs have been saved to:", args.output_file)

