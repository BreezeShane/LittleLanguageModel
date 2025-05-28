"""
Clean data.
"""
# pylint: disable=invalid-name

import jsonlines
from os import getcwd, path, listdir
# import re
from tqdm import tqdm

RAW_DATA_DIR = path.join(getcwd(), "data_process", "raw_data")
CORPUS_DIR = path.join(getcwd(), "corpus")

# def filter_char(text: str, char: str):
#     """ Filter the specify char in text. """
#     return " ".join([
#         x.strip()
#         for x in text.split(char)
#         if x != ""
#     ])
# filtered_text = filter_char(raw_text, "\n").replace("\r", "")
# filtered_text = filter_char(filtered_text, " ")
# filtered_text = filter_char(filtered_text, "\t")

def filter_chars(text: str, chars: list[str]):
    """ Filter chars in the text """
    for char in chars:
        text = text.replace(char, " ")
        if char == "\n":
            text = text.replace("\r", "")
    return " ".join([
        x.strip()
        for x in text.split(" ")
        if x != ""
    ])

for file_name in tqdm(listdir(RAW_DATA_DIR)):
    cur_path = path.join(RAW_DATA_DIR, file_name)
    dest_path = path.join(CORPUS_DIR, "corpus.jsonl")
    with open(cur_path, "r", encoding="utf-8") as f_r:
        raw_text = f_r.read()
        filtered_text = filter_chars(raw_text, ["\n", "\t"])
        # filtered_text = re.sub(r"(.=+.)", " ", string=filtered_text)
        # filtered_text = re.sub(r"(.--+.)", " ", string=filtered_text)

        with jsonlines.open(dest_path, "a") as f_w:
            f_w.write({"text": filtered_text})

        # to_stop = input("Continue? [y] or n")
        # if to_stop == "n":
        #     break


