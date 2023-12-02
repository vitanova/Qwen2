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
    categories = df.columns[1:-1]
    for category in categories:
        prefix = read_in_prompt(category)
        for i_row in range(len(df)):
            human = prefix + df.loc[i_row, 'INPUT']
            assistant = df.loc[i_row, category]
            d_list.append(build_fmt_data(cid, category, human, assistant))
            cid += 1
    writer.write(json.dumps(d_list,ensure_ascii=False))
print(1)
