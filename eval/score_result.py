import json, unicodedata, sacrebleu
import sys

def contains_cj(text):
    for char in text:
        if char == ' ':
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        if not unicodedata.category(char).startswith('L'):
            continue
        if 'CJK' in name or 'KATAKANA' in name or 'HIRAGANA' in name:
            return 1
    return 0

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

def compute_bleu(reference, hypothesis):
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score

def eval_flores(path):
    langs = ['cj', 'latin']
    for lang in langs:
        with open(path, 'r') as f:
            lines = f.readlines()
        cnt = 0
        total = 0
        bleu_arr = []
        for line in lines:
            data = json.loads(line)
            bleu_arr.append(sacrebleu.sentence_bleu(data['query_response'], [data['answer']]).score)
            total += 1
            if lang == 'latin':
                if contains_latin(data['query_response']):
                    cnt += 1
            else:
                if contains_cj(data['query_response']):
                    # print([data['query'] + data['query_response']])
                    cnt += 1
        print(f"{lang} confusion rate: {cnt / total * 100:.2f}", f"BLEU score: {sum(bleu_arr) / len(bleu_arr):.2f}", end=' ')
    

def eval_include(path):
    answer_trans = {
        'Arabic': 'إجابة',
        'Hebrew': 'תשובה',
        'Greek': 'Απάντηση',
        'Korean': '답변',
        'Russian': 'Ответ',
        'Vietnamese': 'Câu trả lời'
    }

    answer_map = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D'
    }
    with open(path, 'r') as f:
        lines = f.readlines()
        cj_cnt = 0
        ok_cnt = 0
        total = 0
        for line in lines:
            data = json.loads(line)
            total += 1
            if contains_cj(data['query_response']):
                cj_cnt += 1
            ok = 0
            response = data['query_response']
            for value in answer_trans.values():
                if f"{value}: {answer_map[data['answer']]}" in response:
                    ok = 1
                    break
            ok_cnt += ok
        print(f"CJ Count: {cj_cnt / total * 100:.2f}", f"Accuracy: {ok_cnt / total * 100:.2f}", end=' ')


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    path = sys.argv[2]
    if dataset_name == 'flores':
        eval_flores(path)
    elif dataset_name == 'include':
        eval_include(path)