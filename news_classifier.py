import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.to(device)

def get_news_sentiments(tokenizer, model, example):
    tokens = tokenizer.encode(example, return_tensors='pt')
    result = model(tokens)
    probabilities = F.softmax(result.logits, dim=1).detach().cpu().numpy().flatten()
    return probabilities


if __name__ == '__main__':
    example = "IRAN TENSION - Iran tested nuclear missiles, and reported that the missile launch was successful, increasing possibility of Canadian military strikes."
    probs = get_news_sentiments(tokenizer, model, example)
    print(probs)