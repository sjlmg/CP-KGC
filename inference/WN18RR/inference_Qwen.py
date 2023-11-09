from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm


def generate_():
    entity2num = {}
    entity2text_dict = {}
    with open('entity2text.txt', 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            num, entity2text = line.strip().split('\t')
            first_comma_index = entity2text.strip().find(',')
            if first_comma_index != -1:
                # 按照第一个逗号分割文本
                split_text = entity2text.split(',', 1)
                entity = split_text[0].strip()  # 第一个部分
                text = split_text[1].strip()  # 第二个部分
                entity2num[num] = entity
                entity2text_dict[num] = text.replace('"', '').replace("'", "")

    base1 = "Give synonyms for '"
    base2 = "' based on the content of the text '"
    base3 = "', and answer in the format {'"
    base4 = "':[your answer]}."
    # temp4 = "Give synonyms for the following entities and return them as a list: "

    # 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
    tokenizer = AutoTokenizer.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

    # 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
    # model = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # 使用CPU进行推理，需要约32GB内存
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
    # 默认使用自动模式，根据设备自动选择精度
    model = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0", device_map="auto",
                                                 trust_remote_code=True).eval()

    # 可指定不同的生成长度、top_p等相关超参
    model.generation_config = GenerationConfig.from_pretrained("/openbayes/input/input0", trust_remote_code=True)

    with open('generate_synonyms_new_Qwen.txt', 'w', encoding='utf-8') as f2:
        count = 0
        for key, value in tqdm(entity2text_dict.items(),desc="processing..."):
            question = base1 + entity2num[key] + base2 + value + base3 + entity2num[key] + base4
            # print(count,question)
            response, history = model.chat(tokenizer, question, history=None)
            response = response.replace("\n", '')
            print(count, response)
            f2.write(entity2num[key])
            f2.write('\t')
            f2.write(response)
            f2.write('\n')
            count += 1


if __name__ == '__main__':
    generate_()

