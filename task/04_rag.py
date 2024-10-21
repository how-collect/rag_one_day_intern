import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from transformers import GPT2Tokenizer
import json 
import pandas as pd 
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

prompt = input("Enter the prompt: ")

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

prompt_vector = model.encode(prompt)

# print(prompt_vector)

vectors = []

with open('json/embedding.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        text = np.array(data.get("text", []))
        vector = np.array(data.get("vector", []))
        if vector is not None:
            vectors.append(vector)

similarities = cosine_similarity([prompt_vector], vectors)[0]

most_similar_index = np.argmax(similarities)

highest_similarity = similarities[most_similar_index]

with open('json/embedding.jsonl', 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i == most_similar_index:
            data = json.loads(line)
            most_similar_text = data.get("text", "")
            break

print(f"Most similar text at index {most_similar_index}:")
print(most_similar_text)

load_dotenv()

openai.api_key = os.getenv("CHATGPT_KEY")

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": most_similar_text},
        {"role": "user", "content": prompt}
    ],
    temperature=1.0
)
response_content = completion.choices[0].message.content.strip()

print("LLM Response: ")
print(response_content)