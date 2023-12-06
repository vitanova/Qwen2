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

# df2 = pd.read_csv('fwd_400plus.csv', encoding='gbk', dtype=str)
# df2 = df2.replace(np.nan, 'nan')
# df2.to_csv('fwd_400plus_utf8.csv', encoding='utf-8', index=False)


df = pd.read_csv('fwd_400plus_utf8.csv', encoding='utf-8', dtype=str)
np.random.seed(20231205)
n_sample = len(df)
n_frac = 0.8
n_train = int(n_sample*n_frac)
idx_train = np.random.choice(n_sample, size=n_train, replace=False)
idx_test = list(set(np.arange(n_sample))-set(idx_train))
df_train = df.loc[idx_train]
df_test = df.loc[idx_test]

df_train.to_csv('fwd_train.csv', index=False, na_rep='nan')
df_test.to_csv('fwd_test.csv', index=False, na_rep='nan')

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


prefix_dict = {
    '方向': """外汇买卖方向分类任务:
下面是一些范例:

美元购汇 -> buy
卖出JPYCNY 2m -> sell
欧元远期 -> nan

请对下述语句进行分类。返回'buy'，'sell'或'nan'。
""",
    '货币': """ISO标准货币代码提取任务:
下面是一些范例:

欧元结汇2y -> eur
明天的汇率 -> nan
buy USD sell JPY 20230401 -> usd jpy

请对下述语句进行分类。依次返回它包含的货币ISO代码或'nan'。
""",
    '外币对方向': """EURUSD买卖分类任务:
下面是一些范例:

buy eurusd 3m -> buy eur sell usd
美元远期 -> nan
卖出欧元买入美元 7m -> sell eur buy usd

请对下述语句进行分类。返回'buy eur sell usd'，'sell eur buy usd'或'nan'。
""", 'TENOR': """外汇标准期限分类任务:
下面是一些范例:

买入3m欧元 -> 3m
明天卖美元买欧元 -> t+1
sell aud t+4 -> nan

请对下述语句进行分类。返回't+0'-'t+3','1w'-'3w', '1m'-'11m','1y'或'nan'。
""", '时间': """日期提取任务:
下面是一些范例:

buy usdcny 2023/06/05 -> 20230605
卖出日元 140.22 20241119 -> 20241119
1m结汇 -> nan
美元2025/22/10结汇 -> nan

请对下述语句进行分类。返回'yyyymmdd'格式的日期或'nan'。
"""
}


def build_fmt_data(cid, category="BUYSELL", human="nan", assistant="nan", dataset="fwd"):
    fmt_data = {"id": f"{cid}",
                "conversations": [{"from": "user", "value": f"{human}"},
                                  {"from": "assistant", "value": f"{assistant}"}]}
    return fmt_data


cid = 0
d_list = []
with open('data.json', mode='w') as writer:
    # categories = df.columns[1:-1]
    categories = df.columns[1:-1]
    for category in categories:
        # prefix = read_in_prompt(category)
        prefix = prefix_dict.get(category)
        for i_row in range(len(df)):
            human = prefix + df.loc[i_row, 'INPUT'] + ' -> '
            assistant = str(df.loc[i_row, category]).lower()
            d_list.append(build_fmt_data(cid, category, human, assistant))
            cid += 1
    writer.write(json.dumps(d_list, ensure_ascii=False))

print(1)
