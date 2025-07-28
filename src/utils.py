import pandas as pd
import spacy
import csv

def safe_field_size_limit(limit):
    try:
        return csv._original_field_size_limit(limit)
    except OverflowError:
        return csv._original_field_size_limit(922337203)

def find_keywords(texts):

    nlp = spacy.load("en_core_web_sm")
    content_pos = ['NOUN', 'PROPN']

    keywords = []
    for text in texts:
        doc = nlp(text)
        words = [token.text for token in doc if token.pos_ == content_pos[1]]
        if len(words) == 0:
            words = [token.text for token in doc if token.pos_ == content_pos[0]]
        keyword = max(set(words), key=words.count)
        keywords.append(keyword)

    return keywords

def extract_provo_texts(data_dir):

    data = pd.read_csv(data_dir, encoding="ISO-8859-1")
    data = pd.DataFrame({'text': [text for text in data['Text'].unique()]})
    data["text"] = data["text"].str.replace("Ñ", "", regex=False)
    data["text"] = data["text"].str.replace("Õ", "", regex=False)
    data['text_id'] = [i+1 for i in range(len(data['text']))]
    data['keyword'] = find_keywords(data["text"].tolist())
    data = data[['text_id', 'keyword', 'text']]

    return data

def extract_meco_texts(data_dir):

    data = pd.read_csv(data_dir)
    data.drop(['Unnamed: 13', 'Unnamed: 14'], axis=1, inplace=True)
    data.columns = ['lang', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    # only English texts
    lan_filter = (data['lang'] == 'English')
    lan_texts_df = data.loc[lan_filter]
    # re-structure data so that each text becomes a row
    trialid_raw_df = lan_texts_df.stack().astype(str).reset_index(level=1)
    trialid_raw_df.rename(columns={'level_1': 'text_id', 0: 'text'}, inplace=True)
    trialid_raw_df = trialid_raw_df.reset_index(drop=False)
    trialid_raw_df.drop([0], inplace=True)
    trialid_raw_df.drop(['index'], axis=1, inplace=True)

    # do some cleaning on each text
    data = trialid_raw_df.copy()
    # replace with "space" the "\\n" at the beginning of a word
    data["text"] = data["text"].str.replace(" \\n", " ", regex=False)
    # replace with "space" the "\\n" between words as "word\\nword"
    data["text"] = data["text"].str.replace("\\n", " ", regex=False)
    # when "word-word" add a space after first word, then the words would be separated equally
    data["text"] = data["text"].str.replace("-", "- ", regex=False)
    # replace with a empty string all the quotation marks
    data["text"] = data["text"].str.replace('"', '', regex=False)

    # add column with keyword (most frequent content word) for each text
    data['keyword'] = find_keywords(data["text"].tolist())
    data = data[['text_id', 'keyword', 'text']]

    return data

def extract_texts(corpus_name:str, data_dir:str='', save_dir:str=''):

    '''
    Extract texts from text file and do some pre-processing in the texts.
    :return: dataframe where each text is a row.
    :param corpus_name: meco or provo
    :param data_dir: filepath to raw data
    :return:
    '''

    if corpus_name == 'provo':
        if not data_dir:
            data_dir = "../data/raw/Provo_Corpus-Predictability_Norms.csv"
        dataset = extract_provo_texts(data_dir)
        if not save_dir:
            save_dir = "../data/processed/provo_texts.csv"
        dataset.to_csv(save_dir, index=False)

    elif corpus_name == 'meco':
        if not data_dir:
            data_dir = "../data/raw/supp_texts.csv"
        dataset = extract_meco_texts(data_dir)
        if not save_dir:
            save_dir = "../data/processed/meco_texts.csv"
        dataset.to_csv(save_dir, index=False)
    else:
        raise NotImplementedError("Parameter `dataset` must be either `provo` or `meco`.")

    return dataset