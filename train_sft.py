import os
import sys
from dataclasses import dataclass
from typing import Optional
from peft import LoraConfig, get_peft_model
import transformers
import trl
from datasets import load_dataset, concatenate_datasets
from swanlab.integration.transformers import SwanLabCallback
import logging

# 添加当前工作目录到系统路径
sys.path.append(os.getcwd())

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))  # 设置日志格式
logger.addHandler(handler)

@dataclass
class DatasetArguments:
    """数据集参数的数据类"""
    dataset_id_or_path: str = "Congliu/Chinese-DeepSeek-R1-Distill-data-110k"  # 数据集 ID 或路径
    dataset_splits: str = "train"  # 数据集拆分
    tokenizer_name_or_path: Optional[str] = None  # 分词器名称或路径
    sampling: int = 20000  # 采样数量

@dataclass
class SwanlabArguments:
    """SwanLab参数的数据类"""
    swanlab: bool  # 是否使用 SwanLab
    workspace: str  # SwanLab 用户名
    project: str  # SwanLab 的项目名
    experiment_name: str  # SwanLab 的实验名

def train():
    # 解析输入参数
    parser = trl.TrlParser((trl.ModelConfig, DatasetArguments, SwanlabArguments, trl.SFTConfig))
    model_args, dataset_args, swanlab_args, training_args = parser.parse_args_and_config()

    # 加载模型
    os.makedirs(training_args.output_dir, exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        dataset_args.tokenizer_name_or_path if dataset_args.tokenizer_name_or_path else model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if "psychology" in dataset_args.dataset_id_or_path:
        # 加载第一个数据集
        dataset1 = load_dataset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k", split=dataset_args.dataset_splits)
        dataset1 = dataset1.shuffle(training_args.seed+100)

        # 加载第二个数据集
        dataset2 = load_dataset(dataset_args.dataset_id_or_path, split=dataset_args.dataset_splits)
        dataset2 = dataset2.shuffle(training_args.seed)

        # 按比例合并数据集
        total_size = dataset_args.sampling  # 使用给定的总数据数量
        dataset1_size = min(int(total_size * 0.2), len(dataset1))
        dataset2_size = min(int(total_size * 0.8), len(dataset2))
        print(f"合并数据集... 总长度为 {total_size}")
        print(f"数据集1的长度: {dataset1_size}, 数据集2的长度: {dataset2_size}")  # 打印数据集的长度
        dataset = concatenate_datasets([dataset1.select(range(dataset1_size)), dataset2.select(range(dataset2_size))])
        print(f"合并后的数据集长度: {len(dataset)}")
    else:
        dataset = load_dataset(dataset_args.dataset_id_or_path, split=dataset_args.dataset_splits)
        dataset = dataset.shuffle(training_args.seed)
        dataset = dataset.select(range(dataset_args.sampling))

    def formatting_prompts_func(example):
        human_text = example["input"]
        gpt_text = example["content"]
        reasoning_data = example["reasoning_content"]
        return f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n<think>{reasoning_data}</think>\n<answer>{gpt_text}</answer><|im_end|>"

    instruction_template = '<|im_start|>user\n'
    response_template = '<|im_start|>assistant\n'

    # SFTTrainer的Collator
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        use_cache=not training_args.gradient_checkpointing,
    )

    callbacks = []
    if swanlab_args.swanlab:
        swanlab_callback = SwanLabCallback(
            workspace=swanlab_args.workspace,
            project=swanlab_args.project,
            experiment_name=swanlab_args.experiment_name,
        )
        callbacks.append(swanlab_callback)

    # PEFT微调模型
    if model_args.use_peft:
        model = get_peft_model(model, LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=model_args.lora_task_type,
        ))
        model.print_trainable_parameters()  # 打印可训练参数的数量

    trainer = trl.SFTTrainer(
        model,
        formatting_func=formatting_prompts_func,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.accelerator.wait_for_everyone()

if __name__ == '__main__':
    train()