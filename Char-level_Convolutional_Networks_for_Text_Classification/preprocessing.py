import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import JamoTokenizer


def read_data(filepath):
    df = pd.read_csv((filepath), sep='\t').loc[:, ['document', 'label']]
    print('original len:{}'.format(len(df)))
    return df


def remove_na(df):
    data = df.loc[df['document'].isna().apply(lambda elm: not elm), :]
    return data


def hist(data):
    tokenizer = JamoTokenizer()
    sen_len = data['document'].apply(lambda sen: len(tokenizer.tokenize(sen)))
    sen_len.hist()
    plt.show()


if __name__ == '__main__':
    with open('./config.json') as io:
        params = json.loads(io.read())

    df = read_data(params['filepath'].get('tr'))
    data = remove_na(df)
    hist(data)
