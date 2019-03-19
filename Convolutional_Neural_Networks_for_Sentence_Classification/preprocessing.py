import pandas as pd
from const import TRAIN_FP, TEST_FP
from konlpy.tag import Okt
import pickle
import gluonnlp as nlp
import itertools


def read_data(filepath):
    df = pd.read_csv((filepath), sep='\t').loc[:, ['document', 'label']]
    print('original len:{}'.format(len(df)))
    return df


def remove_na(df):
    data = df.loc[df['document'].isna().apply(lambda elm: not elm), :]
    return data


def extract_morph(data):
    tokenizer = Okt()
    data['document'] = data['document'].apply(tokenizer.morphs)

    # making the vocab
    counter = nlp.data.count_tokens(itertools.chain.from_iterable([token for token in data['document']]))
    vocab = nlp.Vocab(counter=counter, min_freq=10, bos_token=None, eos_token=None)

    # connecting SISG embedding with vocab
    ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko')
    vocab.set_embedding(ptr_embedding)

    return vocab


def save_vocab(vocab, filepath):
    # saving vocab
    with open(filepath, mode='wb') as io:
        pickle.dump(vocab, io)


if __name__ == '__main__':
    df = read_data(TRAIN_FP)
    tr_data = remove_na(df)
    vocab = extract_morph(tr_data)
    save_vocab(vocab, '../data/nsmc/vocab.pkl')
