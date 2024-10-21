import json 
import pandas as pd 
import numpy as np
import re
import nltk
import spacy
import string
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

file = open('data/Norwegian-Wood.txt','r')

with open('data/Norwegian-Wood.txt', 'r', encoding='utf-8') as file:
    full_text = file.read()

sections = [full_text[i:i+100] for i in range(0, len(full_text), 100)]

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

vectors = model.encode(sections)

json_data = []

for i, (section, vector) in enumerate(zip(sections, vectors)):
    json_data.append({
        "text": section,
        "vector": vector.tolist()
    })

with open('json\embedding.jsonl', 'w', encoding='utf-8') as f:
    for item in json_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')