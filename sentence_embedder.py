import torch
from sentence_transformers import SentenceTransformer


device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def generate_embeddings(sentences):
    return model.encode(sentences)


if __name__ == '__main__':
    sentences = ['This is an example sentence.']
    embeddings = generate_embeddings(sentences)[0]
    print(len(embeddings))

