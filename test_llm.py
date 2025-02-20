import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model(model, tokenizer,  prompt, max_length=8192):  # 添加max_length参数
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length)  # 使用max_length限制生成的长度
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # 跳过特殊token
    
    print("="*10 ,f"\nModel: {model_path}\nResponse: {response}\n")

if __name__ == '__main__':
    prompts = [
        "李白是谁？",
        "我在信任问题上遇到了困难，我该如何解决这些问题",
        "请解释一下量子力学的基本概念。",
        "如何提高学习效率？",
        "我最近一直难以入睡，但不知道原因是什么",
        "我在恋爱关系中正面临着信任问题",
        "我正在与身体形象问题作斗争，这影响了我的心理健康。我可以做些什么来更好地看待自己"
    ]
    

    model_path = "Kedreamix/Xinjing-LM/"


    model = AutoModelForCausalLM.from_pretrained(model_path, 
                        torch_dtype=torch.bfloat16, 
                        attn_implementation="flash_attention_2",
                        device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for prompt in prompts:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>"
        test_model(model, tokenizer, text)