import re

def squash_doc_content(   
            title:str,
            doc_title:str="",
            doc_bte:str="",
            sep:str="\n") -> str:
    return title + sep + doc_title + sep + doc_bte + sep 

def lines_in_file(file_path:str):
    return sum([1 for _ in open(file_path)])


def get_document_stats(text:str):
    # Count words
    words = re.findall(r'\w+', text)
    word_count = len(words)
    
    # Count sentences
    sentences = re.split(r'[.!?]', text)
    # Filter out empty strings resulting from split
    sentence_count = len([sentence for sentence in sentences if sentence.strip() != ''])
    
    return len(text), word_count, sentence_count