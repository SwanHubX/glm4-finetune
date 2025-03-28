"""
simple scripts to remix alpaca data and glavie toolcall data
by ShaohonChen
"""

import json
import datasets


def glaive_trans_data(example):
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
    return {"conversations": conversations}


def glaive_trans_data(example):
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
    return {"conversations": conversations}


def alpaca_trans_data(example):
    """
    process data format (support batch)
    """
    tools = []
    prompt = example["instruction"]
    if len(example["input"]) != 0:
        prompt += "\n\n" + example["input"]
    conversations = [
        {"role": "system", "content": "", "tools": tools},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example["output"]},
    ]

    return {"conversations": conversations}


if __name__ == "__main__":

    glaive_data = datasets.load_dataset(
        "json", data_files="data/glaive_toolcall_zh_1k.json", split="train"
    )
    glaive_data = glaive_data.map(
        glaive_trans_data, remove_columns=glaive_data.column_names
    )
    print(glaive_data)
    alpaca_data = datasets.load_dataset(
        "json", data_files="data/alpaca_gpt4_data_zh.json", split="train"
    ).select(range(1000))
    alpaca_data = alpaca_data.map(
        alpaca_trans_data, remove_columns=alpaca_data.column_names
    )
    print(alpaca_data)

    mix_data = datasets.combine(alpaca_data, glaive_data)
    mix_data.to_json("data/toolcall.json")
