from datasets import load_dataset
import pandas as pd
import json
from tqdm import tqdm
import pandas as pd
import random

def prepare_flores():
    path = 'flores_lang_map.txt'
    code_to_lang = {}
    with open(path, 'r') as f:
        for line in f:
            pairs = [p.strip().replace('`', '') for p in line.split('|')]
            code_to_lang[pairs[2] + pairs[3]] = pairs[4]

    ds = load_dataset('openlanguagedata/flores_plus')
    df = pd.DataFrame(ds['dev'])
    res_df = []

    lang_dict = {}

    for group_name, group_df in tqdm(list(df.groupby('id'))):
        src_en = group_df[group_df['glottocode'] == 'stan1293'].iloc[0]['text']
        src_zh = group_df[group_df['glottocode'] == 'beij1234'].iloc[0]['text']
        for group_row_idx, row in group_df.iterrows():
            for src_lang, src_text in [('English', src_en), ('Chinese', src_zh)]:
                tgt_lang = code_to_lang[row['iso_15924'] + row['glottocode']]
                query = f"Translate the following {src_lang} text into {tgt_lang}.\n{src_lang}: {src_text}\n\n{tgt_lang}:"
                response = row['text']
                lang_dict.setdefault(tgt_lang, []).append({
                    'query': query,
                    'query_response': response,
                })
    random.seed(42)
    for lang, rows in lang_dict.items():
        res_df.extend(random.sample(rows, 100))
    res_df = pd.DataFrame(res_df)
    return res_df

def prepare_alpaca():
    en_ds = load_dataset("yahma/alpaca-cleaned")
    zh_ds = load_dataset("shibing624/alpaca-zh")
    def map_row(row):
        # print(row)
        query = row['instruction'] + row['input']
        query_response = row['output']
        return {
            'query': query,
            'query_response': query_response,
        }
    sampled_en_ds = en_ds['train'].shuffle(seed=42).select(range(10000))
    sampled_zh_ds = zh_ds['train'].shuffle(seed=42).select(range(10000))
    en_rows = [map_row(row) for row in sampled_en_ds]
    zh_rows = [map_row(row) for row in sampled_zh_ds]
    rows = en_rows + zh_rows
    return pd.DataFrame(rows)

def prepare_aya():
    ds = load_dataset("CohereLabs/aya_dataset")
    df = ds['train'].to_pandas()
    sampled_df = df.groupby('language').apply(lambda x: x.sample(n=min(600, len(x)), random_state=42)).reset_index(drop=True)
    def map_row(row):
        # print(row)
        query = row['inputs']
        query_response = row['targets']
        return {
            'query': query,
            'query_response': query_response,
        }
    rows = list(sampled_df.apply(map_row, axis=1))
    return pd.DataFrame(rows)

def prepare_dpsk():
    ds = load_dataset("lightblue/reasoning-multilingual-R1-Llama-70B-train")
    df = ds['train'].to_pandas()
    def map_row(row):
        # print(row)
        query = row['translated_prompt']
        query_response = row['response']
        return {
            'query': query,
            'query_response': query_response,
        }
    rows = list(df.apply(map_row, axis=1))
    return pd.DataFrame(rows)

aya_df = prepare_aya()
dpsk_df = prepare_dpsk()
alpaca_df = prepare_alpaca()
flores_df = prepare_flores()
print('concatenating datasets')
full_df = pd.concat([aya_df, dpsk_df, flores_df, alpaca_df], axis=0)
shuffled_full_df = full_df.sample(frac=1)
print('saving dataset')
shuffled_full_df = shuffled_full_df.to_dict('records')
with open('./data/codeswitch_gate_opensource_alpaca_train.jsonl', 'w', encoding='utf-8') as f:
    for row in tqdm(shuffled_full_df):
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
print('done')