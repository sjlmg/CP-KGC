import torch
from modelscope import Model, snapshot_download
from modelscope.models.nlp.llama2 import Llama2Tokenizer


def generate_():
    entity2num = {}
    entity2text_dict = {}
    with open('entity2text3.txt', 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            num, entity2text = line.strip().split('\t')
            first_comma_index = entity2text.strip().find(',')
            if first_comma_index != -1:
                # 按照第一个逗号分割文本
                split_text = entity2text.split(',', 1)
                entity = split_text[0].strip()  # 第一个部分
                text = split_text[1].strip()  # 第二个部分
                entity2num[num] = entity
                entity2text_dict[num] = text.replace('"', '').replace("'", "")

    base1 = "Give synonyms for '"
    base2 = "' based on the content of '"
    base3 = "'. NOTICE!!!You just need to answer synonyms in python list! No explanation needed! "
    # base4 = "':[your answer]}."
    # temp4 = "Give synonyms for the following entities and return them as a list: "

    model_dir = snapshot_download("modelscope/Llama-2-7b-chat-ms", revision='v1.0.2',
                                  ignore_file_pattern=[r'.+\.bin$'])
    tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
    model = Model.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map='auto')

    with open('generate_synonyms_new_Qwen.txt', 'w', encoding='utf-8') as f2:
        count = 0
        for key, value in entity2text_dict.items():
            question = base1 + entity2num[key] + base2 + value + base3 + entity2num[key]
            # print(count,question)

            system = 'you are a helpful assistant!'
            inputs = {'text': question, 'system': system, 'max_length': 512}
            output = model.chat(inputs, tokenizer)
            print(count, output['response'])

            f2.write(entity2num[key])
            f2.write('\t')
            f2.write(output['response'].replace("\n", ""))
            f2.write('\n')
            count += 1


if __name__ == '__main__':
    generate_()
