"""
Refer: https://huggingface.co/docs/trl/sft_trainer#add-special-tokens-for-chat-format for more advance tools
"""

import argparse
from typing import Optional, Union, List
from dataclasses import dataclass, field
import json

import datasets
from transformers import AutoTokenizer, TrainerCallback
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
import swanlab


################
# Model kwargs
################
@dataclass
class ChatGLM4ModelConfig(ModelConfig):
    model_name_or_path: Optional[str] = field(
        default="./weights/glm-4-9b-hf",
        metadata={
            "help": "Model checkpoint for weights initialization. default used glm4"
        },
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    use_peft: bool = field(
        default=True,
        metadata={"help": "Whether to use PEFT for training. Default true"},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[list[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"],
        metadata={"help": "LoRA target modules."},
    )


################
# Datasets kwargs
################
@dataclass
class DataTrainingArguments:
    data_files: Optional[str] = field(
        default="./data/glaive_toolcall_zh_1k.json",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )


################
# Train kwargs
################
@dataclass
class MySFTConfig(SFTConfig):
    output_dir: Optional[str] = field(
        default="./output/lora-glm4-9b-toolcall",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written. Defaults to 'lora-glm4-9b-toolcall' if not provided."
        },
    )
    num_train_epochs: float = field(
        default=10.0, metadata={"help": "Total number of training epochs to perform."}
    )
    # max_steps: int = field(
    #     default=200,
    #     metadata={
    #         "help": "If > 0: set total number of training steps to perform. Override num_train_epochs."
    #     },
    # )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    learning_rate: float = field(
        default=5e-4, metadata={"help": "The initial learning rate for AdamW."}
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    bf16_full_eval: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may"
                " change."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_seq_length` are truncated "
            "from the right. If `None`, no truncation is applied. When packing is enabled, this value sets the "
            "sequence length."
        },
    )
    eval_strategy: Union[str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: Optional[float] = field(
        default=0.2,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_steps: float = field(
        default=0.2,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )


################
# Print prediction text callback
################
class SavePredictCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps

    def on_save(self, args, state, control, model, processing_class, **kwargs):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_today_weather",
                    "description": "获得某一城市的当日天气",
                    "parameters": {
                        "type": "str",
                        "properties": {
                            "city_name": {"type": "str", "description": "城市名称"}
                        },
                        "required": ["city_name"],
                    },
                },
            },
        ]
        if state.is_world_process_zero:
            tokenizer = processing_class
            batch_test_message = [
                [{"role": "user", "content": "你叫什么名字？"}],
                [
                    {"role": "system", "content": "", "tools": tools},
                    {"role": "user", "content": "告诉我北京的天气怎么样？"},
                ],
            ]
            batch_inputs_text = tokenizer.apply_chat_template(
                batch_test_message,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                padding_side="left",
                add_generation_prompt=True,
            ).to(model.device)

            # print(batch_inputs_text)
            outputs = model.generate(**batch_inputs_text, max_new_tokens=128)
            batch_reponse = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            log_text_list = [swanlab.Text(response) for response in batch_reponse]
            swanlab.log({"Prediction": log_text_list}, step=state.global_step)


def main(model_args, data_args, training_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "[gMASK]<sop>{% for item in messages %}{% if item['tools'] is defined %}<|system|>\n你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具{% set tools = item['tools'] %}{% for tool in tools %}{% if tool['type'] == 'function' %}\n\n## {{ tool['function']['name'] }}\n\n{{ tool['function'] | tojson(indent=4) }}\n在调用上述函数时，请使用 Json 格式表示调用的参数。{% elif tool['type'] == 'python' %}\n\n## python\n\n当你向 `python` 发送包含 Python 代码的消息时，该代码将会在一个有状态的 Jupyter notebook 环境中执行。\n`python` 返回代码执行的输出，或在执行 60 秒后返回超时。\n`/mnt/data` 将会持久化存储你的文件。在此会话中，`python` 无法访问互联网。不要使用 `python` 进行任何网络请求或者在线 API 调用，这些在线内容的访问将不会成功。{% elif tool['type'] == 'simple_browser' %}\n\n## simple_browser\n\n你可以使用 `simple_browser` 工具。该工具支持以下函数：\n`search(query: str, recency_days: int)`：使用搜索引擎进行查询并显示结果，可以使用 `recency_days` 参数控制搜索内容的时效性。\n`mclick(ids: list[int])`：获取一系列指定 id 的页面内容。每次调用时，须选择3-10个页面。选择多个角度的页面，同时尽可能选择可信任的信息来源。考虑到部分页面是无法加载的，你也可以多打开一些可能有用的页面而不用担心内容过多。\n`open_url(url: str)`：打开指定的 URL。\n\n使用 `【{引用 id}†{引用文本}】` 来引用内容。\n\n操作步骤：1. 使用 `search` 来获得信息列表; 2. 使用 `mclick` 来获取指定 ID 页面的内容; 3. 根据获得的内容进行回复。在回复中应当引用信息来源。\n 如果用户提供了 URL，也可以用 `open_url` 直接打开页面。\n如果初次搜索结果没有找到合适的信息，也可以再次使用 `search` 进行搜索。{% elif tool['type'] == 'cogview' %}\n\n## cogview\n\n如果用户的请求中包含了对图像的描述，你可以使用 `cogview` 来生成图像并展示给用户。你需要向 `cogview` 发送图像描述，规则：\n- 发送给 `cogview` 的消息必须使用英语。用户的中文描述必须完全翻译为英语。\n- 应当尽可能详细地描述图像生成的需求，需求描述约 100 英文单词。\n- 保持用户原始描述的意图。不要虚构内容或者没见过的人物。\n- 如无特殊说明，所在地为中国，持有中国立场并遵循中国社会主义价值观。{% endif %}{% endfor %}{% endif %}{% if item['content'] %}<|{{ item['role'] }}|>{{ item['metadata'] }}\n{{ item['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
    ################
    # Dataset
    ################
    raw_datasets = datasets.load_dataset(
        "json", data_files=data_args.data_files, split="train"
    )

    def trans_data(example):
        """
        process data format (support batch)
        """
        tools = []
        try:
            tool_list = json.loads(example["tools"])
        except:
            tool_list = []
        for tool in tool_list:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                },
            )
        conversations = [{"role": "system", "content": "", "tools": tools}]
        for chat in example["conversations"]:
            if chat["from"] == "human":
                role = "user"
            elif chat["from"] == "observation":
                role = "observation"
            else:
                role = "assistant"
            conversations.append({"role": role, "content": chat["value"]})
        return {
            "text": tokenizer.apply_chat_template(
                conversation=conversations, tokenize=False
            )
        }

    text_datasets = raw_datasets.map(
        trans_data, remove_columns=raw_datasets.column_names, num_proc=8
    )
    text_datasets = text_datasets.train_test_split(0.1)  # split train test data

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=None,
        train_dataset=text_datasets["train"],
        eval_dataset=(
            text_datasets["test"] if training_args.eval_strategy != "no" else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=[SavePredictCallback()],
    )
    trainer.train()

    # Save
    trainer.save_model(training_args.output_dir)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ChatGLM4ModelConfig, DataTrainingArguments, MySFTConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "sft", help="Run the SFT training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    model_args, data_args, training_args = parser.parse_args_and_config()
    main(model_args, data_args, training_args)
