import torch
from transformers import GPTJForCausalLM, AutoTokenizer
import pandas as pd
import argparse
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
import nltk
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=1)
args = parser.parse_args()

path = args.model_path

nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTJForCausalLM.from_pretrained(path)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


texts = []
refs = []
jsonObj = pd.read_json(path_or_buf='superni_test_set.jsonl', lines=True)
for i in range(len(jsonObj)):
    texts.append(jsonObj['prompt'][i])
    refs.append(jsonObj['completion'][i])

print(len(texts))
print('generation begins')
generation = []
with torch.no_grad():
    for text in texts:
        encoding = tokenizer(text, padding=True, return_tensors='pt').to(device)
        generated_ids = model.generate(**encoding, max_length=1024)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generation.append(generated_texts[0])
            #sum = []

print('evaluation begins')

R_1, R_2, R_L = 0,0,0


scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
for (ref,text) in zip(refs, texts):
    ref = word_tokenize(ref)
    text = word_tokenize(text)
    scores = scorer.score(ref,text)
    R_1 += scores["rouge1"]
    R_2 += scores["rouge2"]
    R_L += scores["rougeL"]

print(f"The avg rouge-1 score is {R_1/len(refs)}")
print(f"The avg rouge-2 score is {R_2/len(refs)}")
print(f"The avg rouge-L score is {R_L/len(refs)}")

print('writing begins')
# for text in generated_texts:
#   print("---------")
#   print(text)
#name = str(pro)
import pickle
with open(path +'result.pkl', 'wb') as f:
    pickle.dump(generation, f)




