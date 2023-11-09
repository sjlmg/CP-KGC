import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from tqdm import tqdm

def calculate_similarity(words, model, tokenizer):
    word_vectors = []
    for word in words:
        input_ids = tokenizer.encode(word, add_special_tokens=True, max_length=128, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids)
            word_embedding = outputs.last_hidden_state.mean(dim=1)
        word_vectors.append(word_embedding)
    similarity_matrix = []
    for i in range(len(word_vectors)):
        similarity_row = []
        for j in range(len(word_vectors)):
            similarity = cosine_similarity(word_vectors[i].numpy(), word_vectors[j].numpy())[0][0]
            similarity_row.append(similarity)
        similarity_matrix.append(similarity_row)
    return similarity_matrix

def process_synonyms(input_file, output_file, model, tokenizer, threshold=0.9, argmax_synonyms=2):
    with open(input_file, 'r', encoding='utf-8') as f1, open(output_file, 'w', encoding='utf-8') as f2:
        for line in tqdm(f1.readlines(), desc='Writing...'):
            try:
                num, entity, synonyms_sentence = line.strip().split('\t')
                synonyms = [item for item in synonyms_sentence.split('[SEP]') if item]
                synonyms_scores = {}
                for i in synonyms:
                    words = [entity, i]
                    scores = calculate_similarity(words, model, tokenizer)
                    avg_similarity = np.mean(scores)
                    if avg_similarity >= threshold:
                        synonyms_scores[i] = round(avg_similarity, 2)
                sorted_synonyms_scores = OrderedDict(sorted(synonyms_scores.items(), key=lambda item: item[1], reverse=True))
                if len(sorted_synonyms_scores) >= 1:
                    f2.write(f"{num}\t{entity}\t")
                    count = 1
                    for key in sorted_synonyms_scores.keys():
                        if count <= argmax_synonyms and key.capitalize() != entity.capitalize():
                            f2.write(f"{key}[SEP]")
                            count += 1
                    f2.write('\n')
            except Exception as e:
                print(f"Error processing line: {line.strip()}. Error: {e}")

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(r'D:\bert')
    model = BertModel.from_pretrained(r'D:\bert')
    model.eval()
    process_synonyms('Data_Qwen/entity2sysnonyms_Qwen.txt', 'Data_Qwen/entity2synonyms_Qwen_argmax.txt', model, tokenizer)
