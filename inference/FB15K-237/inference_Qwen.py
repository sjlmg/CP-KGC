# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm

# Initialize the tokenizer with special token attack protection disabled by default
tokenizer = AutoTokenizer.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

# Initialize the model with precision and device settings
# Uncomment the appropriate model initialization based on your GPU's capabilities
# For GPUs like A100, H100, RTX3060, RTX3070, etc., enable bf16 precision to save memory
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()

# For GPUs like V100, P100, T4, etc., enable fp16 precision to save memory
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()

# Use CPU for inference, requires about 32GB of memory
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()

# Default to auto mode, which automatically selects precision based on the device
model = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0", device_map="auto", trust_remote_code=True).eval()

# Specify different generation lengths, top_p, and other hyperparameters
model.generation_config = GenerationConfig.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

# Define the prompt for the model
prompt = "Please summarize the following text in one sentence as briefly as possible, and output it in the format {'output':}: "

# Initialize a counter for tracking the number of processed lines
count = 0

# Open the output file for writing the summaries
with open('entity2textlong_summarize.txt', 'w', encoding='utf-8') as f2:
    # Open the input file for reading the texts to be summarized
    with open('entity2textlong.txt', 'r', encoding='utf-8') as f1:
        # Iterate over each line in the input file
        for line1 in tqdm(f1.readlines(), desc="predict..."):
            num, text = line1.strip().split('\t')
            question = prompt + text
            response, history = model.chat(tokenizer, question, history=None)
            print(count, response)

            # Writing num, response, and newline in one line
            f2.write(f"{num}\t{response}\n")

            count += 1
