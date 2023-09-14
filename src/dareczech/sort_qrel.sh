#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 input_file.tsv output_file.tsv"
    exit 1
fi

input_file="$1"
output_file="$2"

if [ ! -f "$input_file" ]; then
    echo "Input file '$input_file' not found."
    exit 1
fi

head -n 1 "$input_file" > $output_file
tail -n +2 "$input_file" | sort -t $'\t' -k2 >> "$output_file"

echo "Sorting complete. Sorted data saved to '$output_file'."
