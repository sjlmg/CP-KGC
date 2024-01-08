# -*- coding:utf-8 -*-
import openai
from tqdm import tqdm

PROMPT = "Please summarize the following text in one sentence as briefly as possible: "
def get_summarized_response(text):
    """Fetch summarized response from GPT-4 for the given text."""
    question = PROMPT + text
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",  # or use gpt3.5-turbo here.
            messages=[
                {"role": "assistant", "content": question}
            ]
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return None, e

def process_text_files():
    """Process entity2textlong.txt and write summarized responses to summarize_GPT4_2.txt."""
    wrong_nums = []

    with open('entity2textlong.txt', 'r', encoding='utf-8') as source_file:
        with open("summarize_GPT4.txt", 'w', encoding='utf-8') as output_file:
            for count, line in enumerate(tqdm(source_file.readlines(), desc="Processing..."), start=1):
                num, text = line.strip().split('\t')
                response, error = get_summarized_response(text)

                if response:
                    print(count, response)
                    output_file.write(f"{num}\t{response}\n")
                else:
                    print(error, num)
                    wrong_nums.append(num)

    with open('wrong_num.txt', 'w', encoding='utf-8') as error_file:
        for num in wrong_nums:
            error_file.write(f"{num}\n")

if __name__ == "__main__":
    process_text_files()
