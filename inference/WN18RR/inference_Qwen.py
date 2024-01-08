# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm


def generate_synonyms():
    # Read and process entity-text pairs from file
    entity2num = {}
    entity2text_dict = {}
    with open('entity2text.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            num, entity2text = line.strip().split('\t')
            first_comma_index = entity2text.find(',')
            if first_comma_index != -1:
                entity, text = entity2text.split(',', 1)
                entity = entity.strip()
                text = text.strip().replace('"', '').replace("'", "")
                entity2num[num] = entity
                entity2text_dict[num] = text

    # Preparing base parts of the query
    query_base1 = "Give synonyms for '"
    query_base2 = "' based on the content of the text '"
    query_base3 = "', and answer in the format {'"
    query_base4 = "':[your answer]}."

    # Initialize the tokenizer with special token attack protection disabled by default
    tokenizer = AutoTokenizer.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

    # Initialize the model. Uncomment the appropriate model initialization based on your GPU's capabilities
    # For bf16 precision (recommended for A100, H100, RTX3060, RTX3070 GPUs) to save memory
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()

    # For fp16 precision (recommended for V100, P100, T4 GPUs) to save memory
    # model = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0", device_map="auto", trust_remote_code=True, fp16=True).eval()

    # For using CPU for inference, requires about 32GB of memory
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()

    # Default to auto mode, which automatically selects precision based on the device
    model = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0", device_map="auto",
                                                 trust_remote_code=True).eval()

    # Configure model generation settings
    model.generation_config = GenerationConfig.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

    # Generate synonyms and write to a new file
    with open('generate_synonyms_new_Qwen.txt', 'w', encoding='utf-8') as output_file:
        for count, (key, value) in enumerate(tqdm(entity2text_dict.items(), desc="processing...")):
            question = f"{query_base1}{entity2num[key]}{query_base2}{value}{query_base3}{entity2num[key]}{query_base4}"
            response, _ = model.chat(tokenizer, question, history=None)
            response_cleaned = response.replace("\n", '')

            print(count, response_cleaned)
            output_file.write(f"{entity2num[key]}\t{response_cleaned}\n")


if __name__ == '__main__':
    generate_synonyms()
