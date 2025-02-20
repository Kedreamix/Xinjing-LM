import os
from threading import Thread
import gradio as gr
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer

# 参数配置
MAX_MAX_NEW_TOKENS = 8192
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# 使用心境大模型（Xinjing-LM）
model_id = "Kedreamix/Xinjing-LM"
trust_remote_code = True
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto",
                                                trust_remote_code=trust_remote_code)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.use_default_system_prompt = False
    
# 生成函数，处理推理过程
def generate(
        message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
) -> str:
    # 强制模型多思考
    prompt_template = f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n<think>"
    
    # 添加历史对话处理
    for user_msg, assistant_msg in chat_history:
        prompt_template = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n<think>{assistant_msg}" + prompt_template
    
    print(prompt_template)
    inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=6.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=inputs.input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # 等待生成结果并返回
    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

# Gradio 界面设置
chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="系统提示", lines=6),
        gr.Slider(
            label="最大生成令牌",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="温度 (Temperature)",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (核采样)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="重复惩罚",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=None,
    examples=[
        ["请分享一些应对焦虑的方法。"],
        ["给我一些缓解压力的建议。"],
        ["如何管理情绪以提升心理健康？"],
        ["有哪些放松的冥想技巧可以帮助减压？"],
        ["我该如何面对负面情绪？"]
    ],
)

# 页面设计
with gr.Blocks(css="style.css") as demo:
    gr.Markdown(
        """<p align="center"><img src="https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png" style="height: 80px"/><p>""")
    gr.Markdown("""<center><font size=8>Xinjing-LM 心理健康助手 👾</center>""")
    gr.Markdown(
        """<center><font size=4>Xinjing-LM 是一款专注于心理健康支持的智能助手，能够提供压力管理、焦虑缓解、情绪调节等方面的建议。</center>""")

    # 渲染ChatInterface
    chat_interface.render()

if __name__ == "__main__":
    demo.queue().launch()