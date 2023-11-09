"""
        对entity2textlong的长文本进行概括（一句话）

        prompt：Please summarize the following text in one sentence as briefly as possible, and output it in the format {'output':}:
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# 默认使用自动模式，根据设备自动选择精度
model = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0", device_map="auto",
                                             trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

prompt = "Please summarize the following text in one sentence as briefly as possible, and output it in the format {'output':}: "

count = 0
with open('entity2textlong_summarize.txt', 'w', encoding='utf-8') as f2:
    with open('entity2textlong.txt', 'r', encoding='utf-8') as f1:
        for line1 in tqdm(f1.readlines(), desc="predict..."):
            num, text = line1.strip().split('\t')
            question = prompt + text
            response, history = model.chat(tokenizer, question, history=None)
            print(count, response)
            f2.write(num)
            f2.write('\t')
            f2.write(response)
            f2.write('\n')
            count += 1