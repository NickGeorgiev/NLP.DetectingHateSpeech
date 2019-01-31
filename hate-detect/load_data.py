import pandas
import random
import json

from preprocessor import initial_text_clean_up


def parse_CSV(csv_path):
    data = pandas.read_csv(csv_path, '\t')
    return [
        (initial_text_clean_up(text),
            "hatespeech" if hs == 1 else "non-hatespeech")
        for (text, hs)
        in list(zip(data['text'], data['HS']))
    ]


if __name__ == '__main__':
    labeled_data = parse_CSV('../data/dev_en.tsv') + parse_CSV('../data/train_en.tsv')
    random.shuffle(labeled_data)
    
    with open('../jsons/labeled_data.json', 'w') as file:
        file.write(json.dumps(labeled_data))

    # TODO: dump features
