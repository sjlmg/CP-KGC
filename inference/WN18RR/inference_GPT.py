# -*- coding: utf-8 -*-
import openai
from tqdm import tqdm

def process_string(input_str):
    """
    有一批字符串的格式为：land_reform_NN_1，主要分为三不部分，分别是land_reform，NN和1，现在需要对这个字符串进行处理，分别获取land reform,NN两部分
    """
    # Splitting the input string by underscores
    parts = input_str.split('_')

    # The first part is the words joined by underscore, replace them with spaces
    words_part = ' '.join(parts[:-2])

    # The second to last part is the tag
    # tag_part = parts[-2]

    return words_part.strip()


def call_with_prompt():

    with open('wordnet-mlj12-definitions_add_example_gpt3.5.txt', 'w', encoding='utf-8') as file:
        with open('data/wordnet-mlj12-definitions_without_examples.txt', 'r', encoding='utf-8') as f3:
            count = 0
            for index,item in tqdm(enumerate(f3.readlines())):
                try:
                    id_,entity_,desc_ = item.strip().split('\t')
                    entity = process_string(entity_)
                    question = (
                                f"input = '{entity}' means '{desc_}', "
                                f"please use the shortest possible text to introduce the usage of '{entity}'.\n"
                                f"output = ")

                    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                              messages=[{"role": "assistant", "content": question}])
                    ans = str(completion.choices[0].message["content"]).replace("\n", "")
                    final_ans = f"{desc_}; {ans}"
                    print(index, ans)
                    file.write(f"{id_}\t{final_ans}\n")
                except:
                    count += 1
                    file.write(f"{id_}\t{desc_}\n")
                break
    print(count)

if __name__ == '__main__':
    call_with_prompt()
