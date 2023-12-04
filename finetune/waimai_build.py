import pandas as pd
import numpy as np
import json

df = pd.read_csv('waimai_10k.csv',
                 encoding='utf-8')
df['label_cn'] = df.apply(lambda row: '好评' if row['label']==1 else '差评',
                          axis=1)
# 只要1000个
df = df.sample(n=1000)
df = df.reset_index().iloc[:, 1:]

prefix = """外卖评论文本分类任务:
下面是一些范例:

味道真不错 -> 好评
太辣了，吃不下都  -> 差评

请对下述评论进行分类。返回'好评'或者'差评'。
"""


def get_prompt(text):
    return prefix + text + ' -> '


def build_fmt_data(cid, category="BUYSELL", human="nan", assistant="nan", dataset="fwd"):
    fmt_data = {"id": f"{cid}",
                "conversations": [{"from": "user", "value": f"{human}"},
                                  {"from": "assistant", "value": f"{assistant}"}]}
    return fmt_data

cid = 0
d_list = []
with open('data2.json', mode='w', encoding='utf8') as writer:
    for i_row in range(len(df)):
        human = prefix + df.loc[i_row, 'review']
        assistant = df.loc[i_row, 'label_cn']
        d_list.append(build_fmt_data(cid, "waimai", human, assistant, 'waimai_10k'))
        cid += 1
    writer.write(json.dumps(d_list, ensure_ascii=False))

print(1)
