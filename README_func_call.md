# 使用ChatGLM4进行大模型工具微调（附代码和测试脚本）

作者：情感机器实验室-陈少宏 邮箱：shaohon_chen@115lab.club

## 摘要

本教程主要实现了一个大模型的工具微调方法。为了便于实现，减少代码量，本文使用了🤗HuggingFace的TRL框架实现。该框架除了支持SFT外，对DPO、PPO、GRPO等流行的强化微调算法都有很好的支持。

虽然使用框架能够极大的减少工作量，但是不可避免的为新手学习带来了困扰。因此本教程会尽量附上完整的文档引用来帮助读者进一步学习框架。诚然从使用pytorch实现微调过程能够极大的提升对过程的理解，社区也有相当多优秀的项目。但是笔者仍推荐大家多使用框架来完成训练，这样可以减少大量的时间来让大家更专注于创新。

因此本教程建议对🤗HuggingFace Transformers框架有一定基础的读者阅读～。

注意：由于ChatGLM的模型相对较大，实际运行大概需要显存>=16G

## 目录

**目录：**



**参考资料：**

* 智谱AI官网：[https://www.zhipuai.cn/](https://www.zhipuai.cn/)

* ChatGLM-9B基座模型：[https://huggingface.co/THUDM/glm-4-9b-hf](https://huggingface.co/THUDM/glm-4-9b-hf/tree/main)

* ChatGLM-9B-Chat模型：[https://huggingface.co/THUDM/glm-4-9b-chat-hf](https://huggingface.co/THUDM/glm-4-9b-chat-hf/tree/main)

* glaive函数调用数据集中文版：[https://huggingface.co/datasets/llamafactory/glaive_toolcall_zh](https://huggingface.co/datasets/llamafactory/glaive_toolcall_zh)

* 本博客开源项目链接：[https://github.com/ShaohonChen/chatglm-finetune](https://github.com/ShaohonChen/chatglm-finetune)

* SwanLab训练日志查看：[https://swanlab.cn/@ShaohonChen/chatglm-finetune/](https://swanlab.cn/@ShaohonChen/chatglm-finetune/)

## TRL包介绍+环境准备

![./docs/trl](./docs/trl.png)

本教程使用[🤗HuggingFace TRL](https://huggingface.co/docs/trl/index)框架来完成微调代码的实现。TRL是一个强大且便于使用的微调框架，除了支持SFT外，也能轻松的通过接口调用DPO、PPO、GRPO等流行的强化微调算法。此外也完美兼容Transformers架构。

首先是安装本教程的环境，安装命令如下：

```bash
pip install transformers trl datasets peft swanlab
```

其中`transformers trl peft`用于模型的加载和训练，`datasets`用于导入数据集，`swanlab`用于对训练过程可视化跟踪。

下面列举一个简单的微调案例来介绍HF TRL框架的使用方法：

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")   # 设置微调数据集，此处使用IMDB电影评论分类数据

training_args = SFTConfig(  # 设置微调参数
    max_length=512,
    output_dir="/tmp",
)
trainer = SFTTrainer(   # 设置模型，此处使用facebook的opt-350M，参数量比较小便于下载
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
)
trainer.train() # 开始训练，流程和TRL一样
```

上面的代码来自HF官方文档[https://huggingface.co/docs/trl/sft_trainer](https://huggingface.co/docs/trl/sft_trainer)，增加了注释便于读者理解。

简单来说TRL包的使用方法和Transformers类似，不过多了两步：

* 导入`SFTConfig`模块，这个模块基于`transformers`的`TrainingArguments`，不过针对SFT引入了一点额外的参数，以及lora的支持参数

* 导入`SFTTrainer`模块，这个模块包含了SFT的代码实现，还有一些对`peft`的lora支持和数据集格式转换代码。

后文将完整的介绍如何使用TRL包完成大模型的函数调用功能。

## ChatGLM4介绍+模型准备

![chatglm_history](./docs/chatglm_history.png)

GLM-4-9B是[智谱AI](https://www.zhipuai.cn/)推出的最新一代预训练模型GLM-4系列中的开源版本。ChatGLM发布了多个版本，其中GLM-4-9B是第四代基座模型，其微调版本GLM-4-9B-Chat具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最大 128K 上下文）等高级功能。

本教程使用GLM-4-9B模型进行函数调用功能微调，并使用SwanLab进行模型的结果跟踪。

⚠️注意：ChatGLM为了配合Huggingface Transformers更新，发布了两个版本权重`THUDM/glm-4-9b`和`THUDM/glm-4-9b-hf`，后者对应更为新版本的transformers，因此本教程使用后者的权重。

本教程以经提供好了下载模型的脚本，下载模型的方法如下：

```bash
huggingface-cli download --local-dir ./weights/glm-4-9b-hf THUDM/glm-4-9b-hf
```

模型将会下载在项目目录下的`./weights/glm-4-9b-hf`中

下面列举一个使用`transformers`加载ChatGLM模型并进行推理的代码：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("THUDM/glm-4-9b-chat-hf").eval().to(device)
inputs = tokenizer.encode("我是ChatGLM，是", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

由于是基座模型，没经过微调，因此模型只会完成`"我是ChatGLM，是"`这段文本的后续补全，运行后会生成如下代码：

```bash
Loading checkpoint shards: 100%|██████████| 4/4 [00:01<00:00,  2.35it/s]
[gMASK]<sop>我是ChatGLM，是人工智能助手。我是ChatGLM，是人工智能助手。我是ChatGLM，是人工智能助手
```

当然上面的例子是一个基座模型推理的例子，该模型只能进行文本生成，如果希望使用对话能力，还是需要加载已经微调好的对话模型，代码如下：

```python
from transformers import pipeline

messages = [
    {"role": "user", "content": "你是谁"},
]
pipe = pipeline("text-generation", model="THUDM/glm-4-9b-chat-hf")
print(pipe(messages))
```

此处我们换了种推理接口，直接使用pipeline完成推理，运行后将会生成如下信息

```bash
Loading checkpoint shards: 100%|██████████| 4/4 [00:01<00:00,  2.24it/s]
Device set to use cuda:0
[{'generated_text': [{'role': 'user', 'content': '你是谁'}, {'role': 'assistant', 'content': '\n我是一个人工智能助手，名为 ChatGLM。我是基于清华大学 KEG 实验室和'}]}]
```

使用`print(model)`将模型的结构打印出来，展示如下：

```text
GlmForCausalLM(
  (model): GlmModel(
    (embed_tokens): Embedding(151552, 4096, padding_idx=151329)
    (layers): ModuleList(
      (0-39): 40 x GlmDecoderLayer(
        (self_attn): GlmAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
          (k_proj): Linear(in_features=4096, out_features=256, bias=True)
          (v_proj): Linear(in_features=4096, out_features=256, bias=True)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        )
        (mlp): GlmMLP(
          (gate_up_proj): Linear(in_features=4096, out_features=27392, bias=False)
          (down_proj): Linear(in_features=13696, out_features=4096, bias=False)
          (activation_fn): SiLU()
        )
        (input_layernorm): GlmRMSNorm((4096,), eps=1.5625e-07)
        (post_attention_layernorm): GlmRMSNorm((4096,), eps=1.5625e-07)
      )
    )
    (norm): GlmRMSNorm((4096,), eps=1.5625e-07)
    (rotary_emb): GlmRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=151552, bias=False)
)
```

可以看到GLM模型的层数达到了惊人的40层😂，因此本身使用Lora进行微调时其可训练参数会比其他模型大一些。

## 数据集准备

数据集我已经提前包括在了github项目当中，可以直接使用如下命令下载完整的实验代码

```bash
git clone https://github.com/ShaohonChen/chatglm-finetune.git
```

如果只想下载数据集，可以直接下载如下文件：

```bash
...
```

## 代码说明+超参数调整

完整的微调代码公开在了GitHub上，使用如下命令即可下载

```bash
git clone https://github.com/ShaohonChen/chatglm-finetune.git
```

文章的附件中也有完整的实现代码[#代码附件](#附件完整代码)

本文接下来重点介绍各个代码的功能模块

加载模型的超参数设置，这里可以重点关注lora参数的设置，本文lora参数参考了ChatGLM官方微调代码的lora参数设置

```python
################
# Model kwargs
################
@dataclass
class ChatGLM4ModelConfig(ModelConfig):
    model_name_or_path: Optional[str] = field(
        default="./weights/glm-4-9b-hf",
        # default="/data/nvme1/weights/Qwen2.5-7B",
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
```

数据集超参数设置，这里比较简单，只是加载了本地的glaive数据集

```python
################
# Datasets kwargs
################
@dataclass
class DataTrainingArguments:
    data_files: Optional[str] = field(
        default="./data/glaive_toolcall_zh_1k.json",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

```

不过为了方便读者理解数据集长什么样，仍旧提供数据集展示脚本

```python
import datasets
raw_dataset=datasets.load_dataset("json", data_files="data/glaive_toolcall_zh_1k.json")
print(raw_dataset)
"""打印内容
DatasetDict({
    train: Dataset({
        features: ['conversations', 'tools'],
        num_rows: 1000
    })
})
"""
```

可以看到数据一共有1000条，并且包括`'conversations', 'tools'`两个字段

进一步选取其中一条打印：

```python
print(raw_dataset["train"][0])
```

输出如下：

```json
{'conversations': [{'from': 'human', 'value': '你好，我需要一个1到100之间的随机数。'},
  {'from': 'function_call',
   'value': '{"name": "generate_random_number", "arguments": {"min": 1, "max": 100}}'},
  {'from': 'observation', 'value': '{"number": 57}'},
  {'from': 'gpt', 'value': '生成的随机数在1到100之间，是57。'},
  {'from': 'human', 'value': '好的，可以。这次生成一个长度在200到300之间的句子。'},
  {'from': 'function_call',
   'value': '{"name": "generate_random_number", "arguments": {"min": 200, "max": 300}}'},
  {'from': 'observation', 'value': '{"number": 267}'},
  {'from': 'gpt', 'value': '生成的随机数在200到300之间，是267。'},
  {'from': 'human', 'value': '谢谢，这些就是我需要的全部。'},
  {'from': 'gpt', 'value': '不客气！如果你还需要其他什么，随时问。'}],
 'tools': '[{"name": "generate_random_number", "description": "在指定范围内生成一个随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值"}, "max": {"type": "integer", "description": "最大值"}}, "required": ["min", "max"]}}]'}
```

可以看出数据集的`conversations`部分和`tools`部分分别定义了模型的问答过程，和能够调用的函数。这里注意，`tools`部分并不总是有能调用的函数，可能出现为空的情况。

ChatGLM提供的推荐输入tools数据结构如下：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "",
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_recommended_books",
            "description": "Get recommended books based on user's interests",
            "parameters": {
              "type": "object",
              "properties": {
                "interests": {
                  "type": "array",
                  "items": {
                    "type": "string"
                  },
                  "description": "The interests to recommend books for"
                }
              },
              "required": [
                "interests"
              ]
            }
          }
        }
      ]
    },
    {
      "role": "user",
      "content": "Hi, I am looking for some book recommendations. I am interested in history and science fiction."
    },
    {
      "role": "assistant",
      "content": "{\"name\": \"get_recommended_books\", \"arguments\": {\"interests\": [\"history\", \"science fiction\"]}}"
    },
    {
      "role": "observation",
      "content": "{\"books\": [\"Sapiens: A Brief History of Humankind by Yuval Noah Harari\", \"A Brief History of Time by Stephen Hawking\", \"Dune by Frank Herbert\", \"The Martian by Andy Weir\"]}"
    },
    {
      "role": "assistant",
      "content": "Based on your interests in history and science fiction, I would recommend the following books: \"Sapiens: A Brief History of Humankind\" by Yuval Noah Harari, \"A Brief History of Time\" by Stephen Hawking, \"Dune\" by Frank Herbert, and \"The Martian\" by Andy Weir."
    }
  ]
}
```

这里可能有一定经验的读者会说，不对呀，我们从0训练我们当然可以定义自己的数据结构。这么想是对的，但是让我们能够直接使用ChatGLM原生的`chat_template`，我还是建议咱们遵守chatglm官方定义的数据格式，这么做的话既能兼容ChatGLM的很多工具，又能充分利用官方定义的special_token。

这里有一个小坑是官方GitHub中对于Tools工具的数据结构是有问题的，因此还是需要将数据集清洗为如上的数据结构！本微调教程的数据集转换函数如下：

```python
def formatting_func(example):
    """
    process data format
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
            role = "users"
        elif chat["from"] == "observation":
            role = "observation"
        else:
            role = "assistant"
        conversations.append({"role": role, "content": chat["value"]})
    return tokenizer.apply_chat_template(conversation=conversations, tokenize=False)
```

这里我们简单打印一下转换完成后数据集最终的一个效果，参考脚本如下：

```python
import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("weights/glm-4-9b-chat-hf")


def formatting_func(example):
    """
    process data format
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
            role = "users"
        elif chat["from"] == "observation":
            role = "observation"
        else:
            role = "assistant"
        conversations.append({"role": role, "content": chat["value"]})
    return tokenizer.apply_chat_template(conversation=conversations, tokenize=False)

print(formatting_func(raw_dataset["train"][0]))
```

输出效果如下，以下字段便是实际运用于模型微调时，输入给模型的数据样式：

```text
[gMASK]<sop><|system|>
你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。

# 可用工具
## generate_random_number

{
    "name": "generate_random_number",
    "description": "在指定范围内生成一个随机数",
    "parameters": {
        "type": "object",
        "properties": {
            "min": {
                "type": "integer",
                "description": "最小值"
            },
            "max": {
                "type": "integer",
                "description": "最大值"
            }
        },
        "required": [
            "min",
            "max"
        ]
    }
}
在调用上述函数时，请使用 Json 格式表示调用的参数。<|users|>
你好，我需要一个1到100之间的随机数。<|assistant|>
{"name": "generate_random_number", "arguments": {"min": 1, "max": 100}}<|observation|>
{"number": 57}<|assistant|>
生成的随机数在1到100之间，是57。<|users|>
好的，可以。这次生成一个长度在200到300之间的句子。<|assistant|>
{"name": "generate_random_number", "arguments": {"min": 200, "max": 300}}<|observation|>
{"number": 267}<|assistant|>
生成的随机数在200到300之间，是267。<|users|>
谢谢，这些就是我需要的全部。<|assistant|>
不客气！如果你还需要其他什么，随时问。
```

最后便是训练的超参数设置和训练过程的实现，这里由于数据规模比较小，我们训练600个steps，每个GPU实际batch大小为1*4：

```python
################
# Train kwargs
################
@dataclass
class MySFTConfig(SFTConfig):
    output_dir: Optional[str] = field(
        default="glm4-9b-toolcall",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written. Defaults to 'glm4-9b-toolcall' if not provided."
        },
    )
    max_steps: int = field(
        default=600,
        metadata={
            "help": "If > 0: set total number of training steps to perform. Override num_train_epochs."
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."},
    )
    gradient_accumulation_steps: int = field(
        default=2,
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
        default=512z,
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
```

训练的流程这块如下,使用HF TRL后流程变得非常简洁。

```python
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=None,
        train_dataset=raw_datasets["train"],
        eval_dataset=(
            raw_datasets["test"] if training_args.eval_strategy != "no" else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        formatting_func=formatting_func,
    )
    trainer.train()

    # Save
    trainer.save_model(training_args.output_dir)
```

## 启动训练+效果评测

启动训练的命令如下：

```bash
python train.py
```

可以看到如下启动信息

![train](docs/train.png)

如果没登录SwanLab可能会弹出登录提示，这里推荐选择1并在[https://swanlab.cn](https://swanlab.cn)完成注册。即可在线查看到训练进展。

登陆命令如下

```bash
swanlab login
```

在线训练看板展示：

[swanlab](docs/swanlab.png)

**多卡实验**

如果你的卡数比较多，推荐使用多卡训练来极大提升训练速度！首先安装huggingface accelerate和deepspeed来方便的开启zero2多卡训练：

```bash
pip install accelerate deepspeed
```

接下来使用如下命令来开启多卡训练（默认8GPU，可更改num_processes参数为实际卡数）：

```bash
accelerate launch --num_processes 8 --config_file configs/zero2.yaml train.py
```

关于zero2的详细设置在`configs/zero2.yaml`中。

**效果对比**

这里我们对比Qwen2.5-7B模型和GLM-4B的微调效果表现



## 附件：完整代码