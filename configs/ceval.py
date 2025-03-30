"""
@author: cunyue
@file: qwen2.5-0.5b.py
@time: 2025/3/9 18:05
@description: 简单、通用的评测配置

魔改自：https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/collections/leaderboard/qwen.py

数据集下载：
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
unzip OpenCompassData-complete-20240207.zip
cd ./data
find . -name "*.zip" -exec unzip "{}" \;

模型下载：
需自己下载到指定位置，注意必须为huggingface格式

python库下载：
pip install -r requiremnets.txt


额外包下载：
git clone git@github.com:open-compass/human-eval.git
cd human-eval && pip install -e .

运行命令：
python run.py config.py --max-num-workers 最大数量，与GPU数量有关
"""

from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate

with read_base():
    ############################## dataset配置 ##############################
    # ---------------------------- 基础测试集（核心能力快速验证） ----------------------------
    # 考试能力（综合知识评估）
    from opencompass.configs.datasets.ceval.ceval_gen import (
        ceval_datasets,
    )  # 中文综合考试，52个学科

    # 语言能力
    from opencompass.configs.datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen import (
        WiC_datasets,
    )  # 词义消歧

    # 知识应用
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen import (
        BoolQ_datasets,
    )  # 二分类问答

    # 文本理解
    from opencompass.configs.datasets.CLUE_C3.CLUE_C3_gen import (
        C3_datasets,
    )  # 中文阅读理解
    from opencompass.configs.datasets.race.race_gen import race_datasets  # 英文阅读理解

    # 逻辑推理
    from opencompass.configs.datasets.CLUE_cmnli.CLUE_cmnli_gen import (
        cmnli_datasets,
    )  # 中文NLI
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets  # 复杂推理任务

    # ---------------------------- 第二阶段扩展（专项能力深入测试） ----------------------------
    from opencompass.configs.datasets.mmlu.mmlu_gen import (
        mmlu_datasets,
    )  # 英文多领域知识
    from opencompass.configs.datasets.drop.drop_gen import (
        drop_datasets,
    )  # 数值推理（需数学解析）
    from opencompass.configs.datasets.siqa.siqa_gen import siqa_datasets  # 社会常识推理

    # ---------------------------- 第三阶段扩展（专业能力验证） ----------------------------
    from opencompass.configs.datasets.humaneval.humaneval_gen import (
        humaneval_datasets,
    )  # 代码生成

    ############################## summarizer配置 ##############################

    # ---------------------------- 基础能力汇总 ----------------------------
    from opencompass.configs.summarizers.groups.ceval import (
        ceval_summary_groups,
    )  # 中文考试
    from opencompass.configs.summarizers.groups.bbh import (
        bbh_summary_groups,
    )  # 复杂推理

    # ---------------------------- 扩展能力汇总 ----------------------------
    from opencompass.configs.summarizers.groups.mmlu import (
        mmlu_summary_groups,
    )  # 综合知识


datasets = sum((v for k, v in locals().items() if k.endswith("_datasets")), [])


other_summary_groups = []
other_summary_groups.append({"name": "Exam", "subsets": ["ceval", "mmlu"]})
other_summary_groups.append({"name": "Language", "subsets": ["WiC"]})
other_summary_groups.append({"name": "Knowledge", "subsets": ["BoolQ"]})
other_summary_groups.append(
    {"name": "Understanding", "subsets": ["C3", "race-middle", "race-high"]}
)
other_summary_groups.append(
    {
        "name": "Reasoning",
        "subsets": ["cmnli", "bbh", "drop", "siqa", "openai_humaneval"],
    }
)
other_summary_groups.append(
    {
        "name": "Overall",
        "subsets": ["Exam", "Language", "Knowledge", "Understanding", "Reasoning"],
    }
)

summarizer = dict(
    dataset_abbrs=[
        "Overall",
        "Exam",
        "Language",
        "Knowledge",
        "Understanding",
        "Reasoning",
        "--------- 考试 Exam ---------",  # category
        # 'Mixed', # subcategory
        "ceval",
        "mmlu",
        "--------- 语言 Language ---------",  # category
        # '字词释义', # subcategory
        "WiC",
        "--------- 知识 Knowledge ---------",  # category
        # '知识问答', # subcategory
        "BoolQ",
        "--------- 理解 Understanding ---------",  # category
        # '阅读理解', # subcategory
        "C3",
        "race-middle",
        "race-high",
        "--------- 推理 Reasoning ---------",  # category
        # '文本蕴含', # subcategory
        "cmnli",
        "siqa",
        "drop",
        # '代码', # subcategory
        "openai_humaneval",
        # '综合推理', # subcategory
        "bbh",
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], []
    ),
)


models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="glm-4-9b-alpaca",
        path="weights/glm-4-9b-chat-hf",
        peft_path="output/lora-glm4-9b-alpaca"
        max_out_len=256,
        batch_size=128,
        run_cfg=dict(num_gpus=1),
    ),
]
