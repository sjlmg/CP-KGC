# -*- coding: utf-8 -*-
import torch
from modelscope import Model, snapshot_download
from modelscope.models.nlp.llama2 import Llama2Tokenizer


def generate_synonyms():
    # Read and process entity-text pairs from file
    entity2num = {}
    entity2text_dict = {}
    with open('entity2text3.txt', 'r', encoding='utf-8') as file:
        for line in file:
            num, entity2text = line.strip().split('\t')
            entity, _, text = entity2text.partition(',')
            entity = entity.strip()
            text = text.strip().replace('"', '').replace("'", "")
            if entity:
                entity2num[num] = entity
                entity2text_dict[num] = text

    # Synonym generation query bases
    query_base1 = "Give synonyms for '"
    query_base2 = "' based on the content of '"
    query_base3 = "'. NOTICE!!!You just need to answer synonyms in python list! No explanation needed! "

    # Download and load the model and tokenizer
    model_dir = snapshot_download("modelscope/Llama-2-7b-chat-ms", revision='v1.0.2',
                                  ignore_file_pattern=[r'.+\.bin$'])
    tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
    model = Model.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map='auto')

    # Generate synonyms and write to file
    with open('generate_synonyms_new_Qwen.txt', 'w', encoding='utf-8') as output_file:
        for count, (num, text) in enumerate(entity2text_dict.items()):
            entity = entity2num[num]
            question = f"{query_base1}{entity}{query_base2}{text}{query_base3}{entity}"
            system_message = 'you are a helpful assistant!'
            inputs = {'text': question, 'system': system_message, 'max_length': 512}
            response = model.chat(inputs, tokenizer)

            print(count, response['response'])
            output_file.write(f"{entity}\t{response['response'].replace('\n', '')}\n")


if __name__ == '__main__':
    generate_synonyms()
