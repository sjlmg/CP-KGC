def load_synonyms_data(file_path):
    synonyms_data = {}
    synonyms_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            if len(line.strip().split('\t')) > 2:
                num, entity, sentence = line.strip().split('\t')
                added_synonyms = sentence.replace('[SEP]', ', ')
                synonyms_data[num] = added_synonyms
                synonyms_list.append(num)
    return synonyms_data, synonyms_list

def merge_entity_text(synonyms_data, synonyms_list, input_file, output_file):
    entity2text_list = []
    with open(input_file, 'r', encoding='utf-8') as file:
        count = 0
        for line in file.readlines():
            num, entity2text = line.strip().split('\t')
            first_comma_index = entity2text.strip().find(',')
            if first_comma_index != -1:
                # 按照第一个逗号分割文本
                split_text = entity2text.split(',', 1)
                entity = split_text[0].strip()  # 第一个部分
                text_origin = split_text[1].strip()

                if num in synonyms_list:
                    text = synonyms_data[num] + text_origin  # 添加同义词信息
                else:
                    text = text_origin
                temp = f"{num}\t{entity}, {text}"
                entity2text_list.append(temp)
                count += 1

    with open(output_file, 'w', encoding='utf-8') as file:
        for line in entity2text_list:
            file.write(line + '\n')

def create_entity_definitions(input_file, output_file, synonyms_file):
    num2text_dict = {}
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            num, entity2text = line.strip().split('\t')
            first_comma_index = entity2text.strip().find(',')
            if first_comma_index != -1:
                # 按照第一个逗号分割文本
                split_text = entity2text.split(',', 1)
                entity = split_text[0].strip()  # 第一个部分
                text = split_text[1].strip()  # 第二个部分
                num2text_dict[num] = text

    num2entity_dict = {}
    with open(synonyms_file, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            num, entity, origin_text = line.strip().split('\t')
            num2entity_dict[num] = entity

    with open(output_file, 'w', encoding='utf-8') as file:
        for key, value in num2entity_dict.items():
            file.write(f"{key}\t{value}\t{num2text_dict.get(key, '')}\n")

if __name__ == '__main__':
    synonyms_data, synonyms_list = load_synonyms_data('Data_Qwen/entity2synonyms_Qwen_argmax.txt')
    merge_entity_text(synonyms_data, synonyms_list, 'Data_Qwen/entity2text.txt', 'entity2textQwen7b.txt')
    create_entity_definitions('entity2textQwen7bT.txt', 'wordnet-mlj12-definitionsQwenT.txt',
                              'Data_Qwen/wordnet-mlj12-definitions.txt')
