from collections import defaultdict
import pandas as pd
import rdata
import spacy
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os
import string
import re


def extract_provo_texts(data_dir):

    """
    Pre-process file with texts from Provo.
    :param data_dir: filepath to raw Provo text file.
    :return: Output dataframe with columns [text_id, text, keyword]
    """

    data = pd.read_csv(data_dir, encoding="ISO-8859-1")
    data["Text"] = data["Text"].str.replace("doesnÕt", "doesn't", regex=False)
    data = pd.DataFrame({'text': [text for text in data['Text'].unique()]})
    data["text"] = data["text"].str.replace("Ñ", "", regex=False)
    data["text"] = data["text"].str.replace("Õ", "", regex=False)
    data["text"] = data["text"].str.replace('"', '', regex=False)
    data['text_id'] = [i for i in range(len(data['text']))]
    data = data[['text_id', 'text']]

    return data

def extract_meco_texts(data_dir:str):

    """
    Pre-process file with texts from MECO.
    :param data_dir: filepath to raw MECO text file.
    :return: Output dataframe with columns [text_id, text, keyword]
    """

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
    trialid_raw_df['text_id'] = [i for i in range(len(trialid_raw_df['text']))]

    # do some cleaning on each text
    data = trialid_raw_df.copy()
    # replace with "space" the "\\n" at the beginning of a word
    data["text"] = data["text"].str.replace(" \\n", " ", regex=False)
    # replace with "space" the "\\n" between words as "word\\nword"
    data["text"] = data["text"].str.replace("\\n", " ", regex=False)
    # when "word-word" add a space after first word, then the words would be separated equally
    data["text"] = data["text"].str.replace("-", "- ", regex=False)
    # replace with an empty string all the quotation marks
    data["text"] = data["text"].str.replace('"', '', regex=False)

    data = data[['text_id', 'text']]

    return data

def extract_onestop_texts(data_dir:str, level:str=''):

    data = pd.read_csv(data_dir)

    data = data[['article_batch','article_id','article_title','paragraph_id','difficulty_level','paragraph']]
    data['paragraph'] = data['paragraph'].str.replace('51-year- old', '51-year-old', regex=False)
    data['paragraph'] = data['paragraph'].str.replace('top- level', 'top-level')
    data['paragraph'] = data['paragraph'].str.replace('e- bicycles', 'e-bicycles')
    data['paragraph'] = data['paragraph'].str.replace('French- Canadian', 'French-Canadian')
    data['paragraph'] = data['paragraph'].str.replace('brand- new', 'brand-new')
    data['paragraph'] = data['paragraph'].str.replace('honey- flavored', 'honey-flavored')
    data['paragraph'] = data['paragraph'].str.replace('100sq- meter', '100sq-meter')
    data['paragraph'] = data['paragraph'].str.replace('credit- card', 'credit-card')
    data = data.drop_duplicates(subset=['article_batch','article_id','paragraph_id','difficulty_level'])
    data = data.loc[~(data['article_id'] == 0)] # remove practice article
    data.sort_values(['article_batch', 'article_id', 'paragraph_id'], inplace=True)
    data['article_id'] = data['article_id'].apply(lambda x: int(x) - 1)
    data['paragraph_id'] = data['paragraph_id'].apply(lambda x: int(x) - 1)

    # generate articles from paragraphs
    if level == 'article':
        article_df = defaultdict(list)
        for article_info, paragraphs in data.groupby(['article_title', 'article_batch', 'article_id', 'difficulty_level']):
            article_text = ' '.join([paragraph for paragraph in paragraphs['paragraph'].tolist()])
            article_df['article_batch'].append(article_info[1])
            article_df['article_id'].append(article_info[2])
            article_df['difficulty_level'].append(article_info[3])
            article_df['article_title'].append(article_info[0])
            article_df['article'].append(article_text)
        data = pd.DataFrame(article_df)

    data.reset_index(drop=True, inplace=True)

    return data

def assign_article_ianum(group):

    group['article_ianum'] = [i for i in range(len(group))]
    return group

def create_words_df(corpus_name, text_df):

    if corpus_name in ['meco', 'provo']:

        trialids, words, word_ids = [], [], []

        for text_id, text in zip(text_df['text_id'].tolist(), text_df['text'].tolist()):
            text_words = text.split()
            words.extend(text_words)
            trialids.extend([text_id for i in range(len(text_words))])
            word_ids.extend([i for i in range(len(text_words))])

        words_df = pd.DataFrame({'text_id': trialids,
                                  'ianum': word_ids,
                                  'ia': words})

    elif corpus_name == 'onestop':

        words_dict = defaultdict(list)
        for article_batch, article_id, article_title, diff_level, paragraph_id, paragraph in zip(text_df['article_batch'].tolist(),
                                                                                     text_df['article_id'].tolist(),
                                                                                     text_df['article_title'].tolist(),
                                                                                     text_df['difficulty_level'].tolist(),
                                                                                     text_df['paragraph_id'].tolist(),
                                                                                     text_df['paragraph'].tolist()):
            words = paragraph.split(' ')

            for i, word in enumerate(words):
                words_dict['article_batch'].append(article_batch)
                words_dict['article_id'].append(article_id)
                words_dict['article_title'].append(article_title)
                words_dict['difficulty_level'].append(diff_level)
                words_dict['paragraph_id'].append(paragraph_id)
                words_dict['paragraph'].append(paragraph)
                words_dict['ianum'].append(i)
                words_dict['ia'].append(word)

        words_df = pd.DataFrame(words_dict)
        words_df.sort_values(by=['article_batch','article_id','difficulty_level','paragraph_id'], inplace=True)
        words_df = (words_df.groupby(['article_batch','article_id','difficulty_level'])
                    .apply(lambda group:assign_article_ianum(group)).reset_index(drop=True))

    else:
        raise NotImplementedError(f'Corpus {corpus_name} not implemented. Choose between meco, provo, and onestop.')

    return words_df

def extract_texts(corpus_name:str, data_filepath:str='', word_level=True, save_dir:str='', onestop_level:str=''):

    '''
    Extract texts from text file and do some pre-processing in the texts.
    :return: dataframe where each text is a row.
    :param corpus_name: meco, provo or onestop
    :param data_dir: filepath to raw data
    :param save_dir: filepath to save processed data
    :param level: 'article' or 'paragraph' if corpus is onestop
    :return:
    '''

    word_dataset = None

    if corpus_name == 'provo':
        if not data_filepath:
            data_filepath = "../data/raw/Provo_Corpus-Predictability_Norms.csv"
        if not save_dir:
            save_dir = "../data/processed"
        text_dataset = extract_provo_texts(data_filepath)
        filepath_texts = f"{save_dir}/{corpus_name}_texts.csv"
        text_dataset.to_csv(filepath_texts, index=False)
        if word_level:
            word_dataset = create_words_df(corpus_name, text_dataset)
            filepath_words = f"{save_dir}/{corpus_name}_words.csv"
            word_dataset.to_csv(filepath_words, index=False)

    elif corpus_name == 'meco':
        if not data_filepath:
            data_filepath = "../data/raw/supp_texts.csv"
        if not save_dir:
            save_dir = "../data/processed"
        text_dataset = extract_meco_texts(data_filepath)
        filepath_texts = f"{save_dir}/{corpus_name}_texts.csv"
        text_dataset.to_csv(filepath_texts, index=False)
        if word_level:
            word_dataset = create_words_df(corpus_name, text_dataset)
            filepath_words = f"{save_dir}/{corpus_name}_words.csv"
            word_dataset.to_csv(filepath_words, index=False)

    elif corpus_name == 'onestop':
        if not data_filepath:
            data_filepath = "../data/raw/ia_Paragraph_ordinary.csv"
        if not save_dir:
            save_dir = "../data/processed"
        text_dataset = extract_onestop_texts(data_filepath, onestop_level)
        filepath_texts = f"{save_dir}/{corpus_name}_texts.csv"
        if onestop_level == 'article':
            filepath_texts = f"{save_dir}/{corpus_name}_articles.csv"
        text_dataset.to_csv(filepath_texts, index=False)
        if word_level:
            word_dataset = create_words_df(corpus_name, text_dataset)
            filepath_words = f"{save_dir}/{corpus_name}_words.csv"
            word_dataset.to_csv(filepath_words, index=False)

    else:
        raise NotImplementedError("Parameter `corpus_name` must be either `provo`, `meco` or `onestop`.")

    return text_dataset, word_dataset

def pre_process_provo_data(filepath):

    """
    Pre-process word-based data from provo.
    Returns: pre-processed data
    """

    df = pd.read_csv(filepath, encoding="ISO-8859-1")

    # select columns
    df = df[['Participant_ID', 'Text_ID', 'Word', 'Word_Number', 'Sentence_Number', 'Word_In_Sentence_Number',
             'Word_Length','IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_DWELL_TIME']]

    # drop nan values
    df.dropna(subset=['Text_ID', 'Word', 'Word_Number','Sentence_Number','Word_In_Sentence_Number'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # starting indexing with at 0
    df['Text_ID'] = df['Text_ID'].apply(lambda x: int(x) - 1)
    df['Word_Number'] = df['Word_Number'].apply(lambda x: int(x) - 1)
    df['Sentence_Number'] = df['Sentence_Number'].apply(lambda x: int(x) - 1)
    df['Word_In_Sentence_Number'] = df['Word_In_Sentence_Number'].apply(lambda x: int(x) - 1)

    # fix error in ianum sequence
    df['Word_Number'] = df.apply(
        lambda x: x['Word_Number'] - 1 if (x['Text_ID'] == 2) & (x['Word_Number'] >= 45) else x['Word_Number'],
        axis=1)
    df['Word_Number'] = df.apply(
        lambda x: x['Word_Number'] - 1 if (x['Text_ID'] == 12) & (x['Word_Number'] >= 19) else x['Word_Number'],
        axis=1)
    df['Word_Number'] = df.apply(
        lambda x: 50 if (x['Text_ID'] == 17) & (x['Word_Number'] >= 2) & (x['Word'] == 'evolution') else x[
            'Word_Number'],
        axis=1)

    # reorder rows
    df.sort_values(by=['Participant_ID','Text_ID','Word_Number'], inplace=True)

    # fix tokenization
    df['Word'] = df.apply(lambda x: 'true' if x['Word'] == 'TRUE' else x['Word'], axis=1)
    df["Word"] = df["Word"].str.replace('"', '')
    df['Word'] = df.apply(lambda x: x['Word'].replace('?',"'") if ('?' in x['Word']) else x['Word'], axis=1)
    df['Word'] = df.apply(lambda x: '90%' if (x['Word'] == '0.9') & (x['Word_Number'] == 44) else x['Word'], axis=1)
    # words missing full stop
    miss_full_stop = []
    for i, rows in df.groupby(['Participant_ID','Text_ID']):
        last_word = rows['Word'].tolist()[-1]
        last_word_id = rows['Word_Number'].tolist()[-1]
        if '.' not in last_word[-1]:
            if i[1] != 54 and last_word_id != 59:
                miss_full_stop.append((i[0],i[1],last_word_id))
    df["Word"] = df.apply(lambda x: x['Word'] + '.' if (x['Participant_ID'], x['Text_ID'], x['Word_Number']) in miss_full_stop else x['Word'], axis=1)

    # rename columns
    df = df.rename(columns={'Word': 'ia',
                            'Word_Number': 'ianum',
                            'Text_ID': 'text_id',
                            'IA_DWELL_TIME': 'total_dur',
                            'IA_FIRST_FIXATION_DURATION': 'first_fix_dur',
                            'IA_FIRST_RUN_DWELL_TIME': 'gaze_dur',
                            'Participant_ID': 'participant_id',
                            'Sentence_Number': 'sentnum',
                            'Word_In_Sentence_Number': 'abs_word_pos',
                            'Word_Length': 'length'})
    return df

def convert_rdm_to_csv(original_filepath):

    converted = rdata.read_rda(original_filepath)
    converted_key = list(converted.keys())[0]
    df = pd.DataFrame(converted[converted_key])
    filepath = original_filepath.replace('rda', 'csv')
    df.to_csv(filepath)

    return filepath

def pre_process_meco_data(filepath):

    """
    Pre-process word-based English data from MECO.
    Returns: pre-processed word-based English data.
    """

    # convert fixation report to csv
    if filepath.endswith('.rda'):
        filepath = convert_rdm_to_csv(filepath)

    df = pd.read_csv(filepath)

    # filter out non-english data
    if 'lang' in df.columns:
        df = df[(df['lang'] == 'en')]

    # removed unnamed columns if existent
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # select columns
    df = df[['uniform_id', 'trialid', 'sentnum', 'ia', 'ianum', 'reread', 'dur', 'reg.in', 'reg.out', 'skip', 'firstrun.dur', 'firstfix.dur']]

    # drop rows with empty word
    df['ia'] = df['ia'].replace(' ', np.nan)
    df = df.dropna(subset=['ia'])
    df = df.reset_index(drop=True)

    # trialid starts at 0
    df['trialid'] = df['trialid'].apply(lambda x: int(x) - 1)

    # re-index words (bcs of dropping rows with empty word)
    df['ianum'] = df['ianum'].apply(lambda x: int(x) - 1)

    # fix error in ianum sequence
    df['ianum'] = df.apply(
        lambda x: x['ianum'] - 1 if (x['ianum'] >= 149)
                                    & (x['trialid'] == 2)
                                    & (x['uniform_id'] in [f'en_{str(p)}' for p in
                                                           [101, 102, 103, 3, 6, 72, 74, 76, 78, 79, 82, 83, 84, 85, 86,
                                                            87, 88, 89, 90, 91, 93, 94, 95, 97, 98, 99]])
        else x['ianum'], axis=1)

    # fix tokenization
    df["ia"] = df["ia"].str.replace('"', '')

    # rename columns
    df = df.rename(columns={'trialid': 'text_id',
                            'firstrun.dur': 'gaze_dur',
                            'firstfix.dur': 'first_fix_dur',
                            'uniform_id': 'participant_id',
                            'dur': 'total_dur',
                            'reg.in': 'reg_in',
                            'reg.out': 'reg_out'})

    return df

def remove_rows(df, conditions):
    """
    ISSUE: fix errorr in onestop data where some words split into two tokens incorrectly.
    PLAN: get rid of unwanted extra token rows,
          shift later word indices so the numbering is aligned.
    OVERALL PLAN: process only inside the specific participant and paragraph.
    :param df: onestop eye movement data in pandas dataframe.
    :param conditions: dict with specific tokenisation info.
    """

    # unpack conditions
    title_guard = conditions.get("article_title", None)
    diff_guard = conditions.get("difficulty_level", None)
    ia_guard = conditions.get("ia", None)

    ia1, ianum1 = conditions["ia1"], conditions["ianum1"]
    ia2, ianum2 = conditions["ia2"], conditions["ianum2"]
    ianum3 = conditions["ianum3"]

    group_cols = ["participant_id", "article_title", "difficulty_level", "paragraph_id"]

    fixed_groups = []

    for _, g in df.groupby(group_cols, sort=False):
        if title_guard is not None and g["article_title"].iloc[0] != title_guard:
            fixed_groups.append(g)
            continue
        if diff_guard is not None and g["difficulty_level"].iloc[0] != diff_guard:
            fixed_groups.append(g)
            continue
        if ia_guard is not None and ia_guard not in str(g["paragraph"].iloc[0]):
            fixed_groups.append(g)
            continue

        # only fix if the artifact is truly present in this participant+paragraph
        has1 = ((g["ia"] == ia1) & (g["ianum"] == ianum1)).any()
        has2 = ((g["ia"] == ia2) & (g["ianum"] == ianum2)).any()

        if not (has1 and has2):
            fixed_groups.append(g)
            continue

        g2 = g.copy()

        # drop the two error rows
        g2 = g2.loc[~((g2["ia"] == ia1) & (g2["ianum"] == ianum1))]
        g2 = g2.loc[~((g2["ia"] == ia2) & (g2["ianum"] == ianum2))]

        # shift later indices
        g2.loc[g2["ianum"] >= ianum3, "ianum"] = g2.loc[g2["ianum"] >= ianum3, "ianum"] - 1

        fixed_groups.append(g2)

    out = pd.concat(fixed_groups, ignore_index=True)
    return out

def pre_process_onestop_data(filepath):

    df = pd.read_csv(filepath)

    df = df[['participant_id',
             'article_batch',
             'article_id',
             'article_title',
             'paragraph_id',
             'paragraph',
             'difficulty_level',
             'IA_ID',
             'IA_LABEL',
             'IA_FIRST_FIXATION_DURATION',
             'IA_FIRST_RUN_DWELL_TIME',
             'IA_DWELL_TIME',
             'word_length_no_punctuation',
             'subtlex_frequency',
             'wordfreq_frequency',
             'gpt2_surprisal',
             'universal_pos'
             ]]

    df = df.rename(columns={'IA_FIRST_RUN_DWELL_TIME': 'gaze_dur',
                            'IA_FIRST_FIXATION_DURATION': 'first_fix_dur',
                            'IA_DWELL_TIME': 'total_dur',
                            'IA_ID': 'ianum',
                            'IA_LABEL': 'ia'})

    # remove practice articles
    df = df.loc[~(df['article_id'] == 0)]

    # ids start at 0
    df['article_id'] = df['article_id'].apply(lambda x: int(x) - 1)
    df['paragraph_id'] = df['paragraph_id'].apply(lambda x: int(x) - 1)
    df['ianum'] = df['ianum'].apply(lambda x: int(x) - 1)

    # fix error in tokenization (inconsistent tokenization across participants)
    df['ia'] = df['ia'].str.replace('culture"".', 'culture".', regex=False)
    df = remove_rows(df, {'ia': 'deep- fried',
                          'ia1': 'deep-',
                          'ianum1': 15,
                          'ia2': 'fried',
                          'ianum2': 16,
                          'ianum3': 17,
                          'difficulty_level': 'Adv',
                          'article_title': 'Insects Could be the Planets Next Food Source'})
    df = remove_rows(df, {'ia': 'Seven-year- old',
                          'ia1': 'Seven-year-',
                          'ianum1': 85,
                          'ia2': 'old',
                          'ianum2': 86,
                          'ianum3': 87,
                          'difficulty_level': 'Adv',
                          'article_title': 'Bangladeshi Organization Delivers a Lesson on Ending Child Labor'})
    df = remove_rows(df, {'ia': 'top- level',
                          'ia1': 'top-',
                          'ianum1': 69,
                          'ia2': 'level',
                          'ianum2': 70,
                          'ianum3': 71,
                          'difficulty_level': 'Ele',
                          'article_title': 'Autumn-Born Children Better at Sports Says Study'})
    df = remove_rows(df, {'ia': '6.30 am;',
                          'ia1': '6.30',
                          'ianum1': 15,
                          'ia2': 'am;',
                          'ianum2': 16,
                          'ianum3': 17,
                          'difficulty_level': 'Ele',
                          'article_title': 'Why You Should Start Work at 10AM'})
    df = remove_rows(df, {'ia': '10-year- olds',
                          'ia1': '10-year-',
                          'ianum1': 49,
                          'ia2': 'olds',
                          'ianum2': 50,
                          'ianum3': 51,
                          'difficulty_level': 'Ele',
                          'article_title': 'Why You Should Start Work at 10AM'})
    df = remove_rows(df, {'ia': 'al- Mamun.',
                          'ia1': 'al-',
                          'ianum1': 29,
                          'ia2': 'Mamun.',
                          'ianum2': 30,
                          'ianum3': 31,
                          'difficulty_level': 'Ele',
                          'article_title': 'Bangladeshi Organization Delivers a Lesson on Ending Child Labor'})
    df = remove_rows(df, {'ia': '100- seat',
                          'ia1': '100-',
                          'ianum1': 90,
                          'ia2': 'seat',
                          'ianum2': 91,
                          'ianum3': 92,
                          'difficulty_level': 'Adv',
                          'article_title': 'Bright Future for Astrotourism'})
    df = remove_rows(df, {'ia': 'e- bicycles',
                          'ia1': 'e-',
                          'ianum1': 63,
                          'ia2': 'bicycles',
                          'ianum2': 64,
                          'ianum3': 65,
                          'difficulty_level': 'Adv',
                          'article_title': 'Can the US Electric Bike Market Get a Jump Start?'})
    df = remove_rows(df, {'ia': 'French- Canadian',
                          'ia1': 'French-',
                          'ianum1': 47,
                          'ia2': 'Canadian',
                          'ianum2': 48,
                          'ianum3': 49,
                          'difficulty_level': 'Adv',
                          'article_title': 'Man Falls Just Short in Patriot Game to be 100% French'})
    df = remove_rows(df, {'ia': 'brand- new',
                          'ia1': 'brand-',
                          'ianum1': 17,
                          'ia2': 'new',
                          'ianum2': 18,
                          'ianum3': 19,
                          'difficulty_level': 'Adv',
                          'article_title': 'Man Falls Just Short in Patriot Game to be 100% French'})
    df = remove_rows(df, {'ia': 'el- Haite',
                          'ia1': 'el-',
                          'ianum1': 107,
                          'ia2': 'Haite',
                          'ianum2': 108,
                          'ianum3': 109,
                          'difficulty_level': 'Adv',
                          'article_title': 'Morocco Poised to Become a Solar Superpower'})
    df = remove_rows(df, {'ia': '51-year- old',
                          'ia1': '51-year-',
                          'ianum1': 62,
                          'ia2': 'old',
                          'ianum2': 63,
                          'ianum3': 64,
                          'difficulty_level': 'Adv',
                          'article_title': 'The Secrets of the Mystery Shopper'})
    df = remove_rows(df, {'ia': 'honey- flavored',
                          'ia1': 'honey-',
                          'ianum1': 45,
                          'ia2': 'flavored',
                          'ianum2': 46,
                          'ianum3': 47,
                          'difficulty_level': 'Ele',
                          'article_title': 'Rwandan Women Whip up Popular Ice Cream Business'})
    df = remove_rows(df, {'ia': '100sq- meter',
                          'ia1': '100sq-',
                          'ianum1': 41,
                          'ia2': 'meter',
                          'ianum2': 42,
                          'ianum3': 43,
                          'difficulty_level': 'Adv',
                          'article_title': "Vienna Named World's Top City for Quality of Life"})
    df = remove_rows(df, {'ia': 'credit- card',
                          'ia1': 'credit-',
                          'ianum1': 33,
                          'ia2': 'card',
                          'ianum2': 34,
                          'ianum3': 35,
                          'difficulty_level': 'Adv',
                          'article_title': "The Greek Island Where Time Is Running Out"})

    return df

def pre_process_eye_data(corpus_name:str, filepath:str):

    if corpus_name == 'meco':
        eye_data = pre_process_meco_data(filepath)
    elif corpus_name == 'provo':
        eye_data = pre_process_provo_data(filepath)
    elif corpus_name == 'onestop':
        eye_data = pre_process_onestop_data(filepath)
    else:
        raise NotImplementedError(f'Corpus {corpus_name} not implemented. Choose between meco, provo, and onestop.')

    return eye_data

def check_alignment(corpus_name:str, words_df: pd.DataFrame, eye_df: pd.DataFrame, level='text'):

    """
    Check alignment between word and fixation dataframes (whether word ids match).
    :param words_df: words dataframe
    :param eye_df: fixation dataframe
    :param level: 'text' (equivalent to paragraph in onestop) or 'article' (if corpus onestop)
    """

    ianum_column_name = 'ianum'
    if level == 'article':
        ianum_column_name = 'article_ianum'

    if 'output_step' in words_df.columns:
        words_df.rename(columns={'output_step': 'ianum'}, inplace=True)
    if 'current_word' in words_df.columns:
        words_df.rename(columns={'current_word': 'ia'}, inplace=True)

    if corpus_name in ['meco','provo']:
        # for each word if and word in eye-movement dataframe, check if it's the same in word dataframe
        for id, data in eye_df.groupby(['participant_id', 'text_id']):
            text_words = words_df[words_df['text_id'] == id[1]]
            for eye_ia, eye_ianum in zip(data['ia'].tolist(), data['ianum'].tolist()):
                assert not text_words[text_words['ianum'] == eye_ianum].empty, (
                    print(f'ianum {eye_ianum} ({id[0]},{id[1]}) in eye mov data not in text data. '))
                assert not text_words[
                    (text_words['ianum'] == eye_ianum) & (text_words['ia'] == eye_ia)].empty, (
                    print(
                        f'Word {eye_ia} with word id {eye_ianum} ({id[0]},{id[1]}) in eye mov data not in text data. '
                        f'In text data, word id {eye_ianum} yields word {text_words[text_words["ianum"] == eye_ianum]["ia"].tolist()[0]}'))

    elif corpus_name == 'onestop':
        for id, data in eye_df.groupby(['participant_id', 'article_batch', 'article_id', 'difficulty_level', 'paragraph_id']):
            text_words = words_df[(words_df['article_batch'] == id[1]) & (words_df['article_id'] == id[2]) & (words_df['difficulty_level'] == id[3]) & (words_df['paragraph_id'] == id[4])]
            for eye_ia, eye_ianum in zip(data['ia'].tolist(), data[ianum_column_name].tolist()):
                # if word_id in eye movemement data does not exist in words data
                assert not text_words[text_words['ianum'] == eye_ianum].empty, (
                    print(f'ianum {eye_ianum} ({id[0]},{id[1]},{id[2]},{id[3]},{id[4]}) in eye mov data not in text data. '))
                # if word_id in eye movement data yields a different word form in words data
                assert not text_words[
                    (text_words['ianum'] == eye_ianum) & (text_words['ia'] == eye_ia)].empty, (
                    print(f'Word {eye_ia} with word id {eye_ianum} ({id[0]},{id[1]},{id[2]},{id[3]},{id[4]}) in eye mov data not in text data. '
                          f'In text data, word id {eye_ianum} yields word {text_words[text_words["ianum"] == eye_ianum]["ia"].tolist()[0]}'))

    else:
        raise NotImplementedError(f'Corpus {corpus_name} not implemented. Choose between meco, provo, and onestop.')

def add_word_frequency(df, corpus_name, frequency_filepath):

    if corpus_name == 'meco':  # we use frequency file from meco corpus
        freq_col_name = 'zipf_freq'
        word_col_name = 'ia_clean'
        frequency_df = pd.read_csv(frequency_filepath, usecols=[freq_col_name, word_col_name])
        if 'lang' in frequency_df.columns:
            frequency_df = frequency_df[frequency_df['lang'] == 'english']
    elif corpus_name == 'provo':  # we use SUBTLEX-UK
        freq_col_name = 'LogFreq(Zipf)'
        word_col_name = 'Spelling'
        frequency_df = pd.read_csv(frequency_filepath, sep='\t',
                                   usecols=[freq_col_name, word_col_name],
                                   dtype={word_col_name: np.dtype(str)})
    else:
        raise NotImplementedError('Frequency resource or corpus not implemented.')

    frequency_col = []
    for word in df['ia'].tolist():
        word = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), str(word)))
        if word.isalpha():
            word = word.lower()
        if word in frequency_df[word_col_name].tolist():
            frequency_col.append(frequency_df[freq_col_name].tolist()[frequency_df[word_col_name].tolist().index(word)])
        else:
            frequency_col.append(None)

    return frequency_col

def calculate_surprisal_values(df: pd.DataFrame,
                               model:GPT2LMHeadModel|LlamaForCausalLM,
                               tokenizer:GPT2Tokenizer|LlamaTokenizer,
                               device:torch.device)->pd.DataFrame:

    """
    # Calculate the surprisal value for each word from corpus texts.
    Args:
        df: dataframe with words from corpus. It contains the text ids, the word ids, and the words.
        model: gpt2 or llama model.
        tokenizer: gpt2 or llama tokenizer.
        device: cuda or cpu.
    Returns: dataframe with surprisal values.

    """

    # lists to save which words in the corpus are multi-tokens in the model
    model_tokens, corpus_tokens = [], []
    # list to save surprisal values
    surprisal_values = []

    for text, rows in df.groupby('text_id'):

        previous_context = ''

        for i, next_word in enumerate(rows['ia'].tolist()):

            if i == 0:
                # first word in text does not have context to compute surprisal
                surprisal_values.append(None)
                previous_context = next_word

            else:
                next_word = ' ' + next_word
                # tokenize next word
                next_word_id = tokenizer(next_word, return_tensors='pt')["input_ids"][0].to(device)

                # to deal with multi-token words
                total_word_surprisal = 0.0
                for i, token_id in enumerate(next_word_id):
                    if tokenizer.decode([token_id]) not in string.punctuation:
                        # tokenize previous context
                        encoded_input = tokenizer(previous_context, return_tensors='pt').to(device)
                        # turn off dropout layers
                        model.eval()
                        output = model(**encoded_input)
                        # logits are scores from output layer of shape (batch_size, sequence_length, vocab_size)
                        logits = output.logits[:, -1, :]
                        # convert raw scores into probabilities (between 0 and 1)
                        probabilities = torch.nn.functional.softmax(logits,
                                                              dim=1)  # softmax transforms the values from logits into percentages
                        next_token_prob = probabilities[0, token_id]
                        next_token_prob = next_token_prob.cpu().detach().numpy()
                        surprisal = -np.log2(next_token_prob)
                        total_word_surprisal += surprisal
                    previous_context += tokenizer.decode([token_id])
                surprisal_values.append(total_word_surprisal)

                # check which words in the corpus are multi-tokens in the model
                if len(next_word_id) > 1:
                    corpus_tokens.append(next_word)
                    model_tokens.append([tokenizer.decode(token_id) for token_id in
                                         next_word_id])

    return surprisal_values, model_tokens, corpus_tokens

def add_word_surprisal(words_df):

    print('Computing word surprisal...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device ', str(device))

    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # compute surprisal values
    surprisal_values, model_tokens, corpus_tokens = calculate_surprisal_values(words_df, model, tokenizer, device)
    words_df['surprisal'] = surprisal_values

    return words_df, model_tokens, corpus_tokens

def assign_sentence(group):

    paragraph = group['paragraph'].iloc[0]
    sentences = re.split(r'\.\s|\."\s|\.”\s|\?\s|!\s|\?”\s', paragraph)
    words = group['ia'].tolist()
    word_to_sentence, word_to_sentence_length = dict(), dict()
    word_index = 0

    for sent_id, sentence in enumerate(sentences):
        start, end = (word_index, word_index + len(sentence.split()))
        while (word_index < len(words) and
               start <= word_index < end):
            word_to_sentence[word_index] = sent_id
            word_to_sentence_length[word_index] = end - start
            word_index += 1

    assert word_index == len(words), print(word_index, len(words))

    group = group.copy()
    group['sent_id'] = group['ianum'].map(word_to_sentence)
    group['sent_length'] = group['ianum'].map(word_to_sentence_length)

    return group

def assign_word_position_in_sentence(group):

    if len(group['ianum'].tolist()) > 1:
        if 'abs_word_pos' not in group.columns:
            abs_word_pos = [i for i in range(len(group['ianum'].tolist()))]
        else:
            abs_word_pos = group['abs_word_pos'].tolist()
        norm_word_pos = (np.array(abs_word_pos) - np.min(abs_word_pos)) / (np.max(abs_word_pos) - np.min(abs_word_pos))
    else:
        abs_word_pos = np.full(len(group['ianum'].tolist()), np.nan)
        norm_word_pos = np.full(len(group['ianum'].tolist()), np.nan)
    group['abs_word_pos'] = abs_word_pos
    group['norm_word_pos'] = norm_word_pos

    return group

def norm_word_pos_in_text(group):

    if len(group['ianum'].tolist()) > 1:
        norm_word_pos = (np.array(group['ianum'].tolist()) - np.min(group['ianum'].tolist())) / (np.max(group['ianum'].tolist()) - np.min(group['ianum'].tolist()))
    else:
        norm_word_pos = np.full(len(group['ianum'].tolist()), np.nan)
    group['norm_ianum'] = norm_word_pos

    return group

def assign_sentence_length(group):

    group['sent_length'] = [len(group) for i in range(len(group))]
    return group

def assign_sentence_frequency(group):

    group['sent_mean_frequency'] = [np.mean(group['frequency'].tolist()) for i in range(len(group))]
    return group

def assign_pos_tag(group, nlp):

    pos_tags = []
    text = ' '.join(group['ia'].tolist())
    doc = nlp(text)
    index = 0

    for word in group['ia'].tolist():
        if doc[index].text == word: # spacy token equals word in corpus
            pos_tag = doc[index].pos_
            index += 1
        elif doc[index].pos_ == 'PUNCT': # if spacy token is a punctuation which , e.g. ("'
            if doc[index +1].pos_ != 'PUNCT':
                pos_tag = doc[index + 1].pos_
                index += 2
            else: # punctuation followed by punctuation, e.g. ), ).
                pos_tag = doc[index + 2].pos_
                index += 3
        else: # if spacy token does not correspond to the word in corpus e.g. ,.'s
            pos_tag = doc[index].pos_
            index += 2 # skip spacy punctuation or suffix that follows word
        pos_tags.append(pos_tag)

    assert len(pos_tags) == len(group['ia'].tolist())
    group['pos_tag'] = pos_tags
    return group

def normalize(values):
    return (np.array(values) - np.min(values)) / (np.max(values) - np.min(values))

def add_variables_to_word_data(words_df, variables, corpus_name, frequency_filepath='', surprisal_filepath='', path_to_save_multi_tokens=''):

    if corpus_name in ['meco', 'provo']:

        if 'length' in variables:
            words_df['length'] = [len(str(word)) for word in words_df['ia'].tolist()]

        if 'frequency' in variables and frequency_filepath:
            words_df['frequency'] = add_word_frequency(words_df, corpus_name, frequency_filepath)

        if 'surprisal' in variables:
            if surprisal_filepath:
                surprisal_df = pd.read_csv(surprisal_filepath)
                words_df = words_df.merge(surprisal_df[['text_id','ianum','ia','surprisal']],
                              how='left', on=['text_id','ianum','ia'])
            else:
                words_df, model_tokens, corpus_tokens = add_word_surprisal(words_df)
                # write out which words in the corpus are multi-tokens in the model
                if path_to_save_multi_tokens:
                    with open(path_to_save_multi_tokens, 'w') as outfile:
                        outfile.write(f'CORPUS_TOKEN\tMODEL_TOKEN\n')
                        for model_token, corpus_token in zip(model_tokens, corpus_tokens):
                            outfile.write(f'{corpus_token}\t{model_token}\n')

        if 'pos_tag' in variables:
            nlp = spacy.load('en_core_web_sm')
            words_df = (words_df.groupby(['text_id'])
                  .apply(lambda group: assign_pos_tag(group, nlp)).reset_index(drop=True))

        if 'sent_length' in variables:
            words_df = (words_df.groupby(['text_id', 'sent_id'])
                  .apply(lambda group: assign_sentence_length(group)).reset_index(drop=True))

        if 'sent_mean_frequency' in variables and 'frequency' in words_df.columns:
            words_df = (words_df.groupby(['text_id', 'sent_id'])
                  .apply(lambda group: assign_sentence_frequency(group)).reset_index(drop=True))

        if 'word_pos' in variables:
            words_df = (words_df.groupby(['text_id', 'sent_id'])
                  .apply(lambda group: assign_word_position_in_sentence(group)).reset_index(drop=True))

        if 'norm_ianum' in variables:
            words_df = (words_df.groupby(['text_id'])
                  .apply(lambda group: norm_word_pos_in_text(group)).reset_index(drop=True))

        if 'norm_sent_id' in variables:
            words_df['norm_sent_id'] = words_df.groupby(['text_id'])['sent_id'].transform(lambda x:normalize(x))

    elif corpus_name == 'onestop':

        if 'sent_info' in variables:
            words_df = (words_df.groupby(['article_title', 'difficulty_level', 'paragraph_id'])
                        .apply(lambda group: assign_sentence(group)).reset_index(drop=True))

        if 'word_pos' in variables and 'sent_id' in words_df.columns:
            words_df = (words_df.groupby(['article_title','difficulty_level','paragraph_id','sent_id'])
                  .apply(lambda group: assign_word_position_in_sentence(group)).reset_index(drop=True))

        if 'norm_ianum' in variables:
            words_df = (words_df.groupby(['article_title', 'difficulty_level', 'paragraph_id'])
                  .apply(lambda group: norm_word_pos_in_text(group)).reset_index(drop=True))

        if 'text_id' in variables:
            words_df['text_id'] = [f'{article_batch}-{article_id}-{difficulty_level}-{paragraph_id}'
                             for article_batch, article_id, difficulty_level, paragraph_id in
                             zip(words_df['article_batch'].tolist(), words_df['article_id'].tolist(),
                                 words_df['difficulty_level'].tolist(), words_df['paragraph_id'].tolist())]

        if 'article_text_id' in variables:
            words_df['article_text_id'] = [f'{article_batch}-{article_id}-{difficulty_level}'
                             for article_batch, article_id, difficulty_level in
                             zip(words_df['article_batch'].tolist(), words_df['article_id'].tolist(),
                                 words_df['difficulty_level'].tolist())]

    else:
        raise NotImplementedError("`corpus_name` must be either `provo`, `meco` or `onestop`.")

    return words_df

def add_variables_to_eye_data(variables:list[str],
                  df:pd.DataFrame,
                  corpus_name:str,
                  words_df:pd.DataFrame=None,
                  frequency_filepath:str='')->pd.DataFrame:

    """
    Add length, frequency and pos-tag to eye-tracking dataframe.
    :param variables: list of possible variables (length, frequency, pos-tag).
    :param df: dataframe with eye-tracking data
    :param corpus_name: name of eye-tracking corpus
    :param words_df: dataframe with words data
    :param frequency_filepath: path to frequency resource

    Returns: dataframe with eye-tracking data and variables added as columns.

    """

    if corpus_name in ['meco', 'provo']:

        if 'length' in variables:
            if corpus_name == 'meco':
                df['length'] = [len(str(word)) for word in df['ia'].tolist()]

        if 'frequency' in variables and frequency_filepath:
            df['frequency'] = add_word_frequency(df, corpus_name, frequency_filepath)

        if 'surprisal' in variables and 'surprisal' in words_df.columns:
            df = pd.merge(df, words_df[['text_id', 'ianum', 'surprisal']], how='left', on=['text_id', 'ianum'])

        if 'sent_length' in variables:
            df = (df.groupby(['participant_id', 'text_id', 'sent_id'])
                  .apply(lambda group: assign_sentence_length(group)).reset_index(drop=True))

        if 'sent_mean_frequency' in variables and 'frequency' in df.columns:
            df = (df.groupby(['participant_id', 'text_id', 'sent_id'])
                  .apply(lambda group: assign_sentence_frequency(group)).reset_index(drop=True))

        if 'word_pos' in variables:
            df = (df.groupby(['participant_id', 'text_id', 'sent_id'])
                  .apply(lambda group: assign_word_position_in_sentence(group)).reset_index(drop=True))

        if 'norm_ianum' in variables:
            df = (df.groupby(['participant_id', 'text_id'])
                  .apply(lambda group: norm_word_pos_in_text(group)).reset_index(drop=True))

        if 'norm_sent_id' in variables:
            df['norm_sent_id'] = df.groupby(['participant_id', 'text_id'])['sent_id'].transform(lambda x:normalize(x))

    elif corpus_name == 'onestop':

        if 'sent_info' in variables:
            if 'sent_id' not in words_df.columns and 'sent_length' not in words_df.columns:
                words_df = (words_df.groupby(['article_title', 'difficulty_level', 'paragraph_id'])
                            .apply(lambda group: assign_sentence(group)).reset_index(drop=True))
            df = pd.merge(df, words_df[
                ['article_title', 'difficulty_level', 'paragraph_id', 'ianum', 'sent_id', 'sent_length']],
                          how='left', on=['article_title', 'difficulty_level', 'paragraph_id', 'ianum'])

        if 'word_pos' in variables and 'sent_id' in df.columns:
            df = (df.groupby(['participant_id','article_title','difficulty_level', 'paragraph_id','sent_id'])
                  .apply(lambda group: assign_word_position_in_sentence(group)).reset_index(drop=True))

        if 'norm_ianum' in variables:
            df = (df.groupby(['participant_id', 'article_title', 'difficulty_level', 'paragraph_id'])
                  .apply(lambda group: norm_word_pos_in_text(group)).reset_index(drop=True))

        if 'article_ianum' in variables:
            if 'article_ianum' in words_df.columns:
                df = df.merge(words_df[['article_batch', 'article_id', 'difficulty_level', 'paragraph_id', 'ianum', 'article_ianum']],
                              how='left', on=['article_batch', 'article_id', 'difficulty_level', 'paragraph_id', 'ianum'])

        if 'text_id' in variables:
            df['text_id'] = [f'{article_batch}-{article_id}-{difficulty_level}-{paragraph_id}'
                                   for article_batch, article_id, difficulty_level, paragraph_id in
                                   zip(df['article_batch'].tolist(), df['article_id'].tolist(),
                                       df['difficulty_level'].tolist(), df['paragraph_id'].tolist())]

        if 'article_text_id' in variables:
            df['article_text_id'] = [f'{article_batch}-{article_id}-{difficulty_level}'
                             for article_batch, article_id, difficulty_level in
                             zip(df['article_batch'].tolist(), df['article_id'].tolist(),
                                 df['difficulty_level'].tolist())]

    else:
        raise NotImplementedError("`corpus_name` must be either `provo`, `meco` or `onestop`.")

    return df

def main():

    """
    Process corpus files.
    Returns: write out processed file
    """

    # corpus name
    corpus_name = 'provo'  # 'meco'  # 'provo' # 'onestop'
    # file with eye-tracking data
    raw_eye_move_filepath = '../data/raw/ia_Paragraph_ordinary.csv' # '../data/raw/ia_Paragraph_ordinary.csv' # '../data/raw/joint_data_trimmed.csv'  # '../data/raw/Provo_Corpus-Eyetracking_Data.csv'
    raw_text_filepath = ''
    # file with word frequency resource if freq not in eye mov data
    frequency_filepath = '' # '../data/raw/wordlist_meco.csv'  # '../data/raw/SUBTLEX_UK.txt'
    surprisal_filepath = '' # f'../data/processed/{corpus_name}_surprisal.csv'
    # filepath to save out pre-processed eye-tracking data
    processed_eye_move_filepath = f'../data/processed/{corpus_name}_eye_mov.csv'
    processed_words_filepath = f'../data/processed/{corpus_name}_words.csv'

    print('Processing corpus texts...')
    texts_df, words_df = extract_texts(corpus_name, raw_text_filepath, word_level=False, onestop_level='article')
    words_df = pd.read_csv(processed_words_filepath)
    words_df = add_variables_to_word_data(words_df,
                                          ['pos_tag'],
                                          corpus_name,
                                          frequency_filepath,
                                          surprisal_filepath)
                                          # processed_words_filepath.replace('.csv','_surprisal_multi_tokens.csv'))
    words_df.to_csv(processed_words_filepath, index=False) #.replace('.csv','_all.csv')

    print('Processing data with eye movements...')
    eye_data = pre_process_eye_data(corpus_name, raw_eye_move_filepath)
    check_alignment(corpus_name, words_df, eye_data)
    eye_data.to_csv(processed_eye_move_filepath, index=False)
    eye_data = pd.read_csv(processed_eye_move_filepath)
    eye_data = add_variables_to_eye_data(['sent_info', 'word_pos', 'norm_ianum', 'article_ianum', 'text_id', 'article_text_id'],
        eye_data, corpus_name, words_df, frequency_filepath)
    eye_data.to_csv(processed_eye_move_filepath, index=False)
    if corpus_name == 'onestop':
        eye_data = pd.read_csv(processed_eye_move_filepath)
        words_df = words_df.merge(eye_data[['text_id','ianum','ia','wordfreq_frequency','gpt2_surprisal','word_length_no_punctuation','universal_pos']],
                                  how='left',
                                  on=['text_id','ianum','ia'])
        words_df = words_df.drop_duplicates()
        words_df = words_df.rename(columns={'gpt2_surprisal':'surprisal','word_length_no_punctuation':'length','wordfreq_frequency':'frequency','universal_pos': 'pos_tag'})
        words_df.to_csv(processed_words_filepath, index=False)

if __name__ == '__main__':
    main()