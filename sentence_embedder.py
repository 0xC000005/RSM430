import torch
from sentence_transformers import SentenceTransformer


device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer('all-mpnet-base-v2', device=device)

def generate_embeddings(model, sentences):
    embeddings = model.encode(sentences)


if __name__ == '__main__':
    sentences = ['This is an example sentence.']
    generate_embeddings(model, sentences)

