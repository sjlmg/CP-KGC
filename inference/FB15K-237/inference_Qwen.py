from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm

def load_tokenizer():
    """加载分词器"""
    return AutoTokenizer.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

def load_model():
    """加载模型"""
    # 自动根据设备选择精度
    return AutoModelForCausalLM.from_pretrained("/openbayes/input/input0", device_map="auto", trust_remote_code=True).eval()

def configure_model_generation(model):
    """配置模型的生成参数"""
    model.generation_config = GenerationConfig.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

def process_input_file(input_file, output_file, model, tokenizer, prompt):
    """处理输入文件并生成输出文件"""
    with open(input_file, 'r', encoding='utf-8') as f_input, open(output_file, 'w', encoding='utf-8') as f_output:
        for count, line in enumerate(tqdm(f_input, desc="Processing...")):
            num, text = line.strip().split('\t')
            question = prompt + text
            response, _ = model.chat(tokenizer, question, history=None)
            f_output.write(f"{num}\t{response}\n")
            print(count, response)

def main():
    tokenizer = load_tokenizer()
    model = load_model()
    configure_model_generation(model)

    prompt = "Please summarize the following text in one sentence as briefly as possible, and output it in the format {'output':}: "
    process_input_file('entity2textlong.txt', 'entity2textlong_summarize.txt', model, tokenizer, prompt)

if __name__ == "__main__":
    main()
