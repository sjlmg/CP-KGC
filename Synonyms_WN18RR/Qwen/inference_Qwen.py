from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
def load_entity_data(file_path):
    entity2num = {}
    entity2text_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            num, entity2text = line.strip().split('\t')
            entity, text = map(str.strip, entity2text.split(',', 1))
            entity2num[num] = entity
            entity2text_dict[num] = text.replace('"', '').replace("'", "")
    return entity2num, entity2text_dict

def generate_synonyms(entity2num, entity2text_dict, output_file):
    base1 = "Give synonyms for '"
    base2 = "' based on the content of '"
    base3 = "', and answer in the format {'"
    base4 = "':[your answer]}."

    tokenizer = AutoTokenizer.from_pretrained("/openbayes/input/input0", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0", device_map="auto",
                                                 trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

    with open(output_file, 'w', encoding='utf-8') as f2:
        for count, (key, value) in enumerate(tqdm(entity2text_dict.items(), desc="predicting...")):
            question = base1 + entity2num[key] + base2 + value + base3 + entity2num[key] + base4
            response, _ = model.chat(tokenizer, question, history=None)
            response = response.replace("\n", '')
            print(count, response)
            f2.write(entity2num[key] + '\t' + response + '\n')

if __name__ == '__main__':
    entity2num, entity2text_dict = load_entity_data('Data_Qwen/entity2text.txt')
    generate_synonyms(entity2num, entity2text_dict, 'Data_Qwen/generate_synonyms_new_Qwen.txt')
