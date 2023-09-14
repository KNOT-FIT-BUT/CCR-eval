import argparse

parser = argparse.ArgumentParser(
    description="Find duplicate URLs in a TSV file and save them to a JSON file."
    )

parser.add_argument(
    "-d", "--input-dir",
    required=True,
    action="store",
    dest="input_dir",
    help="Path to the input TSV file"

)
parser.add_argument(
    "-o", "--output",
    required=True,
    action="store",
    dest="output_file",
    type=str,
    help="Path to the output JSON file"
)
