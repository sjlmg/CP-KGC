# -*- coding: utf-8 -*-
import openai
from tqdm import tqdm

def read_entity2text(file_path):

    entity2num = {}
    entity2text_dict = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            num, entity2text = line.strip().split('\t')
            first_comma_index = entity2text.strip().find(',')
            if first_comma_index != -1:
                entity, text = entity2text.split(',', 1)
                entity2num[num] = entity.strip()
                entity2text_dict[num] = text.strip().replace('"', '').replace("'", "")

    return entity2num, entity2text_dict


def generate_question(entity, text):
    """generate question"""
    return f"Give synonyms for '{entity}' based on the content of the text '{text}', and answer in python list."


def write_to_file(file_path, data):
    """write"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item + '\n')


def generate_synonyms(entity2num, entity2text_dict, output_path, error_path):
    wrong_list = []
    with open(output_path, 'w', encoding='utf-8') as f:
        for num, text in tqdm(entity2text_dict.items(), desc="Predicting..."):
            question = generate_question(entity2num[num], text)
            try:
                completion = openai.ChatCompletion.create(model="gpt-4",
                                                          messages=[{"role": "assistant", "content": question}])
                response = str(completion.choices[0].message["content"]).replace("\n", "")
                f.write(f"{entity2num[num]}\t{response}\n")
            except Exception as e:
                print(e, num)
                wrong_list.append(num)

    write_to_file(error_path, wrong_list)


if __name__ == '__main__':
    entity2num_data, entity2text_data = read_entity2text('entity2text.txt')
    generate_synonyms(entity2num_data, entity2text_data, 'generate_synonyms_new_GPT4_3.txt', 'wrong_num_GPT4.txt')
