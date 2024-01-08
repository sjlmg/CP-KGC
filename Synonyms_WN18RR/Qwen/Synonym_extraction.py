import re
def extract_words(text):
    # Use a regular expression to match words contained within square brackets and separated by [SEP]
    words = re.findall(r"'(.*?)'", text)
    return '[SEP]'.join(words)

def process_entity_synonyms(input_file, entity_file, output_file):
    with open(entity_file, 'r', encoding='utf-8') as entity_file, \
            open(input_file, 'r', encoding='utf-8') as input_file, \
            open(output_file, 'w', encoding='utf-8') as output_file:

        for entity_line, synonyms_line in zip(entity_file, input_file):
            entity_number, entity_text = entity_line.strip().split('\t')
            entity_name = entity_text.split(',', 1)[0]

            entity_Qwen, synonyms_text = synonyms_line.strip().split('\t')
            if synonyms_text[-1] != '}':
                continue
            else:
                synonyms_list = synonyms_text.split(':')[1][:-1].strip()
                synonyms_sep = extract_words(synonyms_list)

                if entity_name == entity_Qwen:
                    output_file.write(f"{entity_number}\t{entity_Qwen}\t{synonyms_sep}\n")


if __name__ == '__main__':
    process_entity_synonyms('Data_Qwen/generate_synonyms_new_Qwen.txt', 'Data_Qwen/entity2text.txt', 'entity2synonyms_QwenT.txt')
