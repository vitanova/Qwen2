import pandas as pd
import numpy as np
import json

df = pd.read_csv('ZXC_Forward_Tuning_Cases.csv',
                 encoding='utf-8',
                 dtype=str)
df['时间'] = df['时间'].replace(np.nan, 'nan')
df['时间'] = df['时间'].astype('str')


def read_in_prompt(component):
    comp_name = f"prompts/prompt_{component}.txt"
    # Open the file in read mode ('r')
    with open(comp_name, 'r') as file:
        # Read the lines of the file into a list of strings
        lines = file.readlines()
    prompt_i = ''.join(lines)
    return prompt_i


def build_fmt_data(cid, category="BUYSELL", human="nan", assistant="nan", dataset="fwd"):
    fmt_data = {"id": f"{cid}",
                "conversations": [{"from": "user", "value": f"{human}"},
                    {"from": "assistant", "value": f"{assistant}"}]}
    return fmt_data


cid = 0
d_list = []
with open('data.json', mode='w') as writer:
    # categories = df.columns[1:-1]
    categories = df.columns[2:3]
    for category in categories:
        # prefix = read_in_prompt(category)
        prefix = """外汇结汇购汇分类任务:
                下面是一些范例:
                
                我想买美元 -> 购汇
                sell JPYCNY 2m -> 结汇
                欧元远期 -> nan
                
                请对下述评论进行分类。返回'购汇'，'结汇'或'nan'。
                """
        for i_row in range(len(df)):
            human = prefix + df.loc[i_row, 'INPUT'] + ' -> '
            assistant = df.loc[i_row, category]
            d_list.append(build_fmt_data(cid, category, human, assistant))
            cid += 1
    writer.write(json.dumps(d_list, ensure_ascii=False))
print(1)
