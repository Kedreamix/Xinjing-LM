import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化两个客户端，每个客户端对应一个API密钥
clients = [
    OpenAI(api_key="XXX", base_url="https://api-inference.modelscope.cn/v1/"),
    OpenAI(api_key="XXX", base_url="https://api-inference.modelscope.cn/v1/"),
]

current_client_index = 0


def inference_with_deepseek_r1(text):
    """使用DeepSeek R1进行推理，并轮换API密钥"""
    global current_client_index
    client = clients[current_client_index]
    current_client_index = (current_client_index + 1) % len(clients)  # 轮换到下一个客户端
    print(f"Question: {text}")
    # try:
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{
            "role": "system", "content": "你是一位专业心理咨询师，需根据来访者的问题描述提供符合专业且富有同理心的建议和指导。",
            "role": "user", "content": text
            }
        ],
        temperature=0.6
    )
    think_process = response.choices[0].message.reasoning_content
    output_r1 = response.choices[0].message.content
    return think_process, output_r1, "success"
    # except Exception as e:
    #     print(f"推理出错: {str(e)}")
    #     return None, None, "error"  # 返回错误信息


def process_item(item):
    """处理单个数据项"""
    think, output_r1, error = inference_with_deepseek_r1(item['input'])
    # if error:
    #     return {"input": item['input'], "error": error}
    return {
        "input": item['input'],
        "think": think,
        "content": output_r1
    }

def process_dataset_in_reverse(input_file, output_file, error_file, limit=1000, num_threads=4):
    """处理整个数据集，但最多只处理前limit条记录，从最后一条记录开始倒序处理"""
    
    # 读取已处理的输入，避免重复处理
    processed_inputs = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f_out:
            for line in f_out:
                result = json.loads(line)
                processed_inputs.add(result['input'])
            
    except FileNotFoundError:
        pass  # 如果输出文件不存在，则跳过

    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    limit = min(limit, len(dataset))
    
    with open(output_file, 'a', encoding='utf-8') as f_out, open(error_file, 'a', encoding='utf-8') as f_err:
        processed_count = 0
        
        # 使用线程池进行并行处理
        with ThreadPoolExecutor(max_workers=num_threads) as executor:  # 设置并发线程数
            futures = {executor.submit(process_item, item): item for item in dataset[:limit] if item['input'] not in processed_inputs}  # 提交任务
            
            for future in as_completed(futures):
                processed_count += 1
                item = futures[future]
                try:
                    result = future.result()
                    if 'error' in result:
                        f_err.write(json.dumps(result, ensure_ascii=False) + "\n")  # 写入源数据
                        f_err.flush()
                        print(f"处理第 {processed_count}/{limit} 条时出错: {result['error']}")
                    else:
                        json_str = json.dumps(result, ensure_ascii=False)
                        f_out.write(json_str + "\n")  # 修正为每条记录单独写入
                        f_out.flush()
                        
                        print(f"已处理并保存 {processed_count}/{limit} 条")
                except Exception as e:
                    print(f"处理第 {processed_count}/{limit} 条时出错: {str(e)}")
                    f_err.write(json.dumps({"input": item['input'], "error": str(e)}, ensure_ascii=False) + "\n")
                    f_err.flush()

    print("推理完成！")

if __name__ == "__main__":
    process_dataset_in_reverse("translated_dataset10k.json", "inferred_dataset10k.json", "error_log.json", limit=100000, num_threads=8)