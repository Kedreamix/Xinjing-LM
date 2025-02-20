import json
from openai import OpenAI
import time
import concurrent.futures
import random

clients = [
    OpenAI(api_key="XXX", base_url="https://api-inference.modelscope.cn/v1/"),
    OpenAI(api_key="XXX", base_url="https://api-inference.modelscope.cn/v1/"),
]
current_client_index = 0

def translate_text(text):
    """使用OpenAI API进行翻译"""
    # client = random.choice(clients)  # 随机选择一个客户端
    global current_client_index
    client = clients[current_client_index]
    current_client_index = (current_client_index + 1) % len(clients)  # 轮换到下一个客户端
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {
                    "role": "system",
                    "content": "将以下英文内容准确翻译为中文，保持专业术语和心理学术语的正确性，并且最好是问题性的，不以问号结尾，按照以下格式输出: 翻译后的文本:<翻译后的文本>"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3,
            # timeout = 3,
        )
        # 使用正则表达式提取翻译后的文本
        import re
        translated_content = response.choices[0].message.content
        match = re.search(r'翻译后的文本:\s*(.*)', translated_content)
        if match:
            text = match.group(1).strip()
            if text[0] == '<' and text[-1] == '>':
                return text[1:-1]
            return match.group(1).strip()
        else:
            # print("警告：未找到标准翻译格式，返回原始响应")
            return translated_content
    except Exception as e:
        # print(f"翻译出错: {str(e)}")
        pass
        return None


def process_item(item):
    """处理单个数据项并返回翻译结果"""
    translated_input = translate_text(item['input'])
    # 构建包含翻译后内容的字典
    translated_item = {
        "input": translated_input,
    }
    return translated_item
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_dataset(input_file, output_file, error_file, limit=1000, num_threads=4):
    print(input_file)
    """处理整个数据集，但最多只处理前limit条记录"""
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = []
        for line in f:
            try:
                line = line.replace("'input'", "\"input\"").replace("'instruction'", "\"instruction\"").replace("'output'", "\"output\"").replace("'}","\"}").replace("instruction\": '", "instruction\": \"").replace("output\": '", "output\": \"").replace("input\": \'", "input\": \"")
                line = line.replace("', \"instruction", "\", \"instruction").replace("', \"output", "\", \"output")
                # print(len(line))
                dataset.append(json.loads(line))
            except Exception as e:
                print(f"解析错误: {str(e)}，行内容: {line.strip()}")
    limit = min(limit, len(dataset))
    
    with open(output_file, 'a', encoding='utf-8') as f_out, open(error_file, 'a', encoding='utf-8') as f_err:  # 提前打开输出文件
        processed_count = 0  # 计数器，用于跟踪已处理的数据条目数量
        
        # 使用线程池进行并行处理
        with ThreadPoolExecutor(max_workers=num_threads) as executor:  # 设置并发线程数
            futures = {executor.submit(process_item, item): item for item in dataset[:limit]}  # 提交任务
            
            for future in as_completed(futures):
                processed_count += 1
                item = futures[future]
                try:
                    translated_item = future.result()
                    if translated_item['input'] is None:
                        print(f"处理第 {processed_count}/{limit} 条时出错")
                        f_err.write(str(item) + '\n')  # 写入错误样本
                        f_err.flush()
                        continue
                    # 立即写入已翻译的数据
                    json_str = json.dumps(translated_item, ensure_ascii=False)
                    f_out.write(json_str + '\n')
                    f_out.flush()  # 强制刷新缓冲区，确保数据被写入文件

                    print(f"已处理并保存 {processed_count}/{limit} 条")
                except Exception as e:
                    print(f"处理第 {processed_count}/{limit} 条时出错: {str(e)}")
                    f_err.write(str(item) + '\n')  # 写入错误样本
                    f_err.flush()

    print("翻译完成！")


if __name__ == "__main__":
    process_dataset("output_file.json", "translated_dataset10k.json", "error_samples.json", limit=1000000)