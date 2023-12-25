import torch
from modelscope import Model, snapshot_download
from modelscope.models.nlp.llama2 import Llama2Tokenizer

def load_model():
    """加载模型"""
    model_dir = snapshot_download("modelscope/Llama-2-7b-chat-ms", revision='v1.0.2',
                                  ignore_file_pattern=[r'.+\.bin$'])
    tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
    return Model.from_pretrained(model_dir, torch_dtype=torch.float16, device_map='auto'), tokenizer

def parse_input_file(file_path):
    """解析输入文件，提取实体和文本信息"""
    entity2num = {}
    entity2text_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            num, entity2text = line.strip().split('\t')
            first_comma_index = entity2text.strip().find(',')
            if first_comma_index != -1:
                entity, text = entity2text.split(',', 1)
                entity2num[num] = entity.strip()
                entity2text_dict[num] = text.strip().replace('"', '').replace("'", "")
    return entity2num, entity2text_dict

def generate_synonyms(entity2num, entity2text_dict, model, tokenizer, output_file):
    """生成同义词并写入输出文件"""
    base1 = "Give synonyms for '"
    base2 = "' based on the content of '"
    base3 = "'. NOTICE!!!You just need to answer synonyms in python list! No explanation needed! "
    
    with open(output_file, 'w', encoding='utf-8') as f2:
        for count, (key, value) in enumerate(entity2text_dict.items()):
            question = base1 + entity2num[key] + base2 + value + base3 + entity2num[key]
            inputs = {'text': question, 'system': 'you are a helpful assistant!', 'max_length': 512}
            output = model.chat(inputs, tokenizer)
            print(count, output['response'])
            f2.write(f"{entity2num[key]}\t{output['response'].replace('\n', '')}\n")

def main():
    model, tokenizer = load_model()
    entity2num, entity2text_dict = parse_input_file('entity2text3.txt')
    generate_synonyms(entity2num, entity2text_dict, model, tokenizer, 'generate_synonyms_new_Qwen.txt')

if __name__ == '__main__':
    main()
