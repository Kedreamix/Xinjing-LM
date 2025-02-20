import json
from datasets import Dataset

# 打开并加载 JSON 数据
with open("psychology-10k-r1.json", "r", encoding="utf-8") as f:
    data = []
    for line in f:
        try:
            data.append(json.loads(line.strip()))  # 加载每行的 JSON 对象
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# 打印部分数据来检查字段
print(data[:3])  # 查看前三条数据，确认字段名

# 创建数据字典，确保处理字段缺失
data_dict = {
    "input": [item["input"] for item in data],
    "content": [item["content"] for item in data],
    "reasoning_content": [item.get("think", "") for item in data]  # 映射 "think" 字段到 "reasoning_content"
}

# 创建 Hugging Face Dataset
dataset = Dataset.from_dict(data_dict)
# 打印数据集确认
print(dataset)

# 保存为 JSON 文件（如果需要）
dataset.to_json("distill_psychology-10k-r1.json", force_ascii=False)
