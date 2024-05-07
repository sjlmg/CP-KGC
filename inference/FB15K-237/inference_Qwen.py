from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat-GPTQ-Int4")

with open('FB15k_mid2description_qwen-7-int4.txt', 'w', encoding='utf-8') as file:
    with open('data/FB15k_mid2description.txt', 'r', encoding='utf-8') as f3:
        for index,item in tqdm(enumerate(f3.readlines())):
            id_, desc_ = item.strip().split('\t')
            prompt = (f"Task: summarize text, please answer in English.\n"
                        f"Please summarize the following sentence in the shortest possible text while retaining sufficient semantic information.\n"
                        f"sentence = {desc_}")
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
            file.write(f"{id_}\t{ans}\n")
            break
