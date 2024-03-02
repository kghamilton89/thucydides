from dotenv import load_dotenv
from mistralai.client import MistralClient, ChatCompletionResponse

import faiss
import numpy as np
import os
import requests
import sys

load_dotenv()

file_path = 'files/pelo-war.mb.txt'

api_key = os.environ.get("MISTRAL_API_KEY")
pc_api_key = os.environ.get("PINECONE_API_KEY")

if api_key:
    print("Mistral API Key has been added.")
else:
    print("Please add Mistral API Key to continue.")
    sys.exit()

if pc_api_key:
    print("Pinecone API Key has been added.")
else:
    print("Please add Pinecone API Key to continue.")
    sys.exit()

client = MistralClient(api_key)

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

print("Text successfully segmented into" + len(chunks) + "chunks.")

def get_text_embedding(input):
    embeddings_batch_response = client.embeddings(
          model="mistral-embed",
          input=input
      )
    return embeddings_batch_response.data[0].embedding

text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])

d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)