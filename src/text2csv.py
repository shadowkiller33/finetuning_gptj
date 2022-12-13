import csv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train_file_name", type=str, default=1)
parser.add_argument("--test_file_name", type=str, default=1)
args = parser.parse_args()

train_name = args.train_file_name
test_name = args.test_file_name
import pandas as pd
jsonObj = pd.read_json(path_or_buf= str(train_name)+'.jsonl', lines=True)
with open(str(train_name)+'.txt','w', encoding='utf-8') as f:
    for i in range(len(jsonObj)):
        f.write(jsonObj['prompt'][i] + jsonObj['completion'][i])
with open(str(train_name)+'.txt', encoding='utf-8') as txtfile:
    all_text = txtfile.read()
with open(str(train_name)+'.csv', mode='w', encoding='utf-8') as csv_file:
    fieldnames = ['text']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'text': all_text})

jsonObj = pd.read_json(path_or_buf= str(test_name)+'.jsonl', lines=True)
with open(str(test_name)+'.txt','w', encoding='utf-8') as f:
    for i in range(len(jsonObj)):
        f.write(jsonObj['prompt'][i] + jsonObj['completion'][i])
with open(str(test_name)+'.txt', encoding='utf-8') as txtfile:
    all_text = txtfile.read()
with open(str(test_name)+'.csv', mode='w', encoding='utf-8') as csv_file:
    fieldnames = ['text']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'text': all_text})

print("created train.csv and validation.csv files")
