from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
device = "cuda" # the device to load the model onto

def process_string(input_str):
    """
    有一批字符串的格式为：land_reform_NN_1，主要分为三部分，分别是land_reform，NN和1，现在需要对这个字符串进行处理，分别获取land reform,NN两部分
    """
    # Splitting the input string by underscores
    parts = input_str.split('_')

    # The first part is the words joined by underscore, replace them with spaces
    words_part = ' '.join(parts[:-2])

    # The second to last part is the tag
    # tag_part = parts[-2]

    return words_part.strip()

model = AutoModelForCausalLM.from_pretrained(
    "/openbayes/input/input0",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/openbayes/input/input0")

with open('wordnet-mlj12-definitions_add_example_qwen_7_int4.txt', 'w', encoding='utf-8') as file:
    with open('wordnet-mlj12-definitions_without_examples.txt', 'r', encoding='utf-8') as f3:
        for index,item in tqdm(enumerate(f3.readlines())):
            id_, entity_, desc_ = item.strip().split('\t')
            entity = process_string(entity_)
            prompt = (
                        f"input = '{entity}' means '{desc_}', "
                        f"please use the shortest possible text to introduce the usage of '{entity}'.\n"
                        f"output = ")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            print(prompt)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=128
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            ans = response.replace("\n", "")
            print(index,ans)
            final_ans = f"{desc_}; {ans}"
            file.write(f"{id_}\t{final_ans}\n")
            break
