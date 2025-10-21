path = 'flores_lang_map.txt'

with open(path, 'r') as f:
    for line in f:
        pairs = [p.strip().replace('`', '') for p in line.split('|')]
        print(pairs)