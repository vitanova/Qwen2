import pandas as pd
import numpy as np
import json

# df = pd.read_csv('ZXC_Forward_Tuning_Cases.csv',
#                  encoding='utf-8',
#                  dtype=str)
# df['时间'] = df['时间'].replace(np.nan, 'nan')
# df['时间'] = df['时间'].astype('str')
#
# df2 = pd.read_csv('fwd_buysell.csv', encoding='gbk')
# df2.to_csv('fwd_buysell_utf8.csv', encoding='utf-8', index=False)

df = pd.read_csv('fwd_buysell_utf8.csv', encoding='utf-8')
np.random.seed(20231205)
n_train = 300
n_sample = len(df)
idx_train = np.random.choice(n_sample, size=n_train, replace=False)
idx_test = list(set(np.arange(n_sample))-set(idx_train))
df_train = df.loc[idx_train]
df_test = df.loc[idx_test]

df_train.to_csv('fwd_train.csv', index=False)
df_test.to_csv('fwd_test.csv', index=False)

df = df_train
df = df.reset_index().iloc[:, 1:]


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
    categories = df.columns[1:2]
    for category in categories:
        # prefix = read_in_prompt(category)
        prefix = """外汇买卖方向分类任务:
下面是一些范例:

美元购汇 -> buy
卖出JPYCNY 2m -> sell
欧元远期 -> nan

请对下述语句进行分类。返回'buy'，'sell'或'nan'。
"""
        for i_row in range(len(df)):
            human = prefix + df.loc[i_row, 'INPUT'] + ' -> '
            assistant = str(df.loc[i_row, category]).lower()
            d_list.append(build_fmt_data(cid, category, human, assistant))
            cid += 1
    writer.write(json.dumps(d_list, ensure_ascii=False))
print(1)
