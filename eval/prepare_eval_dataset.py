from datasets import load_dataset
import json
from tqdm import tqdm
import pandas as pd
import unicodedata

def contains_latin(text):
    for char in text:
        if char == ' ':
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        if not unicodedata.category(char).startswith('L'):
            continue
        if 'LATIN' in name:
            return 1
    return 0



def pair_df(df, devtest_df, en_map, zh_map, code_to_lang):
    res_df = []
    for _, row in tqdm(df.iterrows()):
        row_id = row['id']
        # src_en = devtest_df[(devtest_df['id'] == row_id) & (devtest_df['glottocode'] == 'stan1293')].iloc[0]['text']
        # src_zh = devtest_df[(devtest_df['id'] == row_id) & (devtest_df['glottocode'] == 'beij1234')].iloc[0]['text']
        src_en = en_map[row_id]
        src_zh = zh_map[row_id]
        # for src_lang, src_text in [('English', src_en), ('Chinese', src_zh)]:
        for src_lang, src_text in [('English', src_en)]:
            tgt_lang = code_to_lang[row['iso_15924'] + row['glottocode']]
            query = f"Translate the following {src_lang} text into {tgt_lang}.\n{src_lang}: {src_text}\n\n{tgt_lang}:"
            response = row['text']
            res_df.append({
                'query': query,
                'query_response': response,
            })
    return res_df


def main():
    ds = load_dataset('openlanguagedata/flores_plus')
    devtest_df = ds['devtest'].to_pandas()

    has_latin_id_arr = []
    no_latin_id_arr = []
    for _, group in devtest_df.groupby('id'):
        langs = ['Hang', 'Arab', 'Hebr', 'Thai', 'Hans']
        has_latin = False
        for lang in langs:
            row = group[group['iso_15924'] == lang].iloc[0]
            if contains_latin(row['text']) == 1:
                has_latin = True
                break
        if has_latin:
            has_latin_id_arr.append(int(group['id'].iloc[0]))
        else:
            no_latin_id_arr.append(int(group['id'].iloc[0]))

    print(len(has_latin_id_arr), len(no_latin_id_arr))
    print(has_latin_id_arr[:10])
    print(no_latin_id_arr[:10])

    glot_arr = ['stan1318', 'hebr1245', 'kore1280', 'thai1261']
    iso_arr = ['Arab', 'Hang', 'Thai', "Hebr"]
    df_without_latin = devtest_df[devtest_df['glottocode'].isin(glot_arr) & devtest_df['iso_15924'].isin(iso_arr) & devtest_df['id'].isin(no_latin_id_arr)]
    df_with_latin = devtest_df[devtest_df['glottocode'].isin(glot_arr) & devtest_df['iso_15924'].isin(iso_arr) & devtest_df['id'].isin(has_latin_id_arr)]

    path = '../train/flores_lang_map.txt'
    code_to_lang = {}
    with open(path, 'r') as f:
        for line in f:
            pairs = [p.strip().replace('`', '') for p in line.split('|')]
            code_to_lang[pairs[2] + pairs[3]] = pairs[4]

    en_map = {row['id']: row['text'] for _, row in devtest_df[devtest_df['glottocode'] == 'stan1293'].iterrows()}
    zh_map = {row['id']: row['text'] for _, row in devtest_df[devtest_df['glottocode'] == 'beij1234'].iterrows()}

    no_latin_df = pair_df(df_without_latin, devtest_df, en_map, zh_map, code_to_lang)
    with_latin_df = pair_df(df_with_latin, devtest_df, en_map, zh_map, code_to_lang)

    with open('./data/flores_no_latin_eval.jsonl', 'w', encoding='utf-8') as f:
        for row in no_latin_df:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    with open('./data/flores_with_latin_eval.jsonl', 'w', encoding='utf-8') as f:
        for row in with_latin_df:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()