import pandas as pd
import json
import re
import os

def assign_sentence(group):

    text = ' '.join(group['ia'].tolist())
    sentences = re.split(r'\.\s|\."\s|\.”\s|\?\s|!\s|\?”\s|;\s', text)
    sentences = [sentence for sentence in sentences if sentence != '']
    words = group['ia'].tolist()
    word_index = 0
    word_to_sentence = dict()

    for sent_id, sentence in enumerate(sentences):
        start, end = (word_index, word_index + len(sentence.split()))
        while (word_index < len(words) and
               start <= word_index < end):
            word_to_sentence[word_index] = sent_id
            word_index += 1

    assert word_index == len(words), print(word_index, len(words))

    group = group.copy()
    group['sent_id'] = group['ianum'].map(word_to_sentence)

    return group

def find_span_indices(span:str, word_list:list[str]):
    """
    Find the start and end indices of a span in a list of words.
    :param span: mention in triplet
    :param word_list: words in eye movement data
    :return: indices of the span in the word list
    """

    span_chars = set(span.lower())
    # print(span_chars)
    best_match = None
    best_candidate = None
    best_score = -1
    for start in range(len(word_list)):
        for end in range(start + 1, len(word_list) + 1):
            candidate = ' '.join(word_list[start:end])
            candidate_chars = set(candidate.lower())
            # print(candidate_chars)
            # [c for c in span_chars if c in candidate_chars]
            score = len((span_chars & candidate_chars))/((len(candidate_chars) + len(span_chars))/2)
            # print(score)
            if score > best_score:
                best_score = score
                best_match = list(range(start, end))
                best_candidate = candidate

    return best_match


def align_triplets_to_words(text_triplets:list[dict], words_df:pd.DataFrame, corpus):
    """
    Given a list of triplets and a list of words, align the triplets to the words based on the triplet mentions.
    :return: dict with word ids as keys and dict with word form and triplets as values
    """
    # does not account for re-analysis (e.g. new triplets extracted from sentence n-1 do not get added to words in sentence n-1 nor do they replace the triplets in n-1)

    word_to_triplet = dict()
    function_words = {'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below'}

    # create dictionary mapping word ids to triplets
    for word, word_id in zip(words_df['ia'].tolist(), words_df['ianum'].tolist()):
        word_to_triplet[str(word_id)] = {'word': word, 'triplets': []}

    # make sure words and triplets were extracted from the same sentence
    for sentence_triplets in text_triplets:
        # match triplet to word, if all the other triplet elements are found in the same sentence
        words_df_filtered = words_df[words_df['sent_id'] <= sentence_triplets['step']]
        # find word indices of mentions in triplet
        mention_indices = []
        for triplet in sentence_triplets['triplets']:
            words = words_df_filtered['ia'].tolist()
            for key, value in triplet.items():
                # entity_1, relation, entity_2
                if key != 'confidence':
                    # for each mention, find the indices in the word list
                    indices = find_span_indices(value['mention'], words)
                    if not indices:
                        if corpus == 'onestop':
                            print(words_df['article_batch'].tolist()[0], words_df['article_id'].tolist()[0], words_df['difficulty_level'].tolist()[0], words_df['paragraph_id'].tolist()[0])
                        elif corpus in ['meco','provo']:
                            print(words_df['text_id'].tolist()[0])
                        for word, sent_id in zip(words_df['ia'],words_df['sent_id']):
                            print(word, sent_id)
                        raise ValueError(f"Mention '{value['mention']}' in triplet {triplet} and step {sentence_triplets['step']} not in word list: {words}")
                    mention_indices.append(indices)
            # find which mention is the latest in the text
            max_index = 0
            latest_mention_index = None
            for mention_index in mention_indices:
                if mention_index:
                    if mention_index[-1] >= max_index:
                        max_index = mention_index[-1]
                        latest_mention_index = mention_index
            # find which word in mention span gets the triplet
            if latest_mention_index:
                # find last content word
                mention_words = words[latest_mention_index[0]:latest_mention_index[-1] + 1]
                content_words = [word for word in mention_words if word.lower() not in function_words]
                if content_words:
                    last_content_word = content_words[-1]
                    last_content_word_id = len(words) - 1 - words[::-1].index(last_content_word)
                    # add triplet only if last content word of last mention is in the last sentence of current step
                    if words_df_filtered['sent_id'].tolist()[last_content_word_id] == sentence_triplets['step']:
                        # add triplet to all following words
                        for word_id in word_to_triplet.keys():
                            if int(word_id) >= int(last_content_word_id):
                                if triplet not in word_to_triplet[word_id]['triplets']:
                                    word_to_triplet[word_id]['triplets'].append(triplet)

    return word_to_triplet

def compile_output_gpt(triplets:dict, words_df:pd.DataFrame, corpus=''):
    """
    :return: dictionary mapping word ids to triplets
    """

    triplet_map = dict()

    if corpus == 'onestop':
        for text_metadata in triplets:
            text_triplets = text_metadata['extracted_triplets']
            article_batch = text_metadata['article_batch']
            article_id = text_metadata['article_id']
            difficulty_level = text_metadata['difficulty_level']
            paragraph_id = text_metadata['paragraph_id']
            # filter words data for the current text
            words_df_filtered = words_df[(words_df['article_batch'] == article_batch) &
                                       (words_df['article_id'] == article_id) &
                                       (words_df['difficulty_level'] == difficulty_level) &
                                       (words_df['paragraph_id'] == paragraph_id)]
            word_to_triplet = align_triplets_to_words(text_triplets, words_df_filtered,corpus)
            triplet_map[f'{article_batch}-{article_id}-{difficulty_level}-{paragraph_id}'] = word_to_triplet

    elif corpus in ['meco','provo']:
        words_df['text_id'] = words_df['text_id'].astype(str)
        for text_metadata in triplets:
            text_triplets = text_metadata['extracted_triplets']
            text_id = str(text_metadata['text_id'])
            # filter words data for the current text
            words_df_filtered = words_df[(words_df['text_id'] == text_id)]
            word_to_triplet = align_triplets_to_words(text_triplets, words_df_filtered,corpus)
            triplet_map[text_id] = word_to_triplet
    else:
        raise NotImplementedError(f'Corpus {corpus} not implemented')

    return triplet_map

def add_triplets_to_words_df(triplet_map:dict, words_df:pd.DataFrame, corpus:str):

    all_rows = []

    if triplet_map:

        for i, row in words_df.iterrows():

            if corpus in ['meco','provo']:
                key= str(row['text_id'])
            elif corpus=='onestop':
                key = f"{row['article_batch']}-{row['article_id']}-{row['difficulty_level']}-{row['paragraph_id']}"
            else:
                raise NotImplementedError(f'Corpus {corpus} not implemented')

            word_id = str(row['ianum'])

            if key in triplet_map.keys():
                triplets = triplet_map.get(key, {}).get(word_id, {}).get('triplets', [])
                new_row = row.to_dict()
                new_row['triplets'] = triplets
                new_row['n_triplets'] = len(triplets)
                all_rows.append(new_row)

        return pd.DataFrame(all_rows)

    else:
        raise ValueError(f'Triplet mapping is empty.')


CORPUS = 'provo'
MODEL = 'gpt-4o-mini'
N_RUNS = 10

# read eye movement word data
words_df = pd.read_csv(f'../data/processed/{CORPUS}_words.csv')
eye_df = pd.read_csv(f'../data/processed/{CORPUS}_eye_mov.csv')

# generate sentence ids if not in dataframe
if CORPUS in ['meco','provo']:
    if 'sent_id' not in words_df.columns:
        words_df = (words_df.groupby(['text_id'])
                    .apply(lambda group: assign_sentence(group)).reset_index(drop=True))
        words_df.to_csv(f'../data/processed/{CORPUS}_words.csv', index=False)

# iterate over runs
datasets = []
for run in range(1, N_RUNS+1):

    output_eye_run_filepath = f'../data/output/{MODEL}/{CORPUS}/{CORPUS}_eye_mov_plus_triplets_{MODEL}_{run}.csv'

    if os.path.exists(output_eye_run_filepath):
        df = pd.read_csv(output_eye_run_filepath)
        datasets.append(df)

    else:

        output_word_run_filepath = f'../data/output/{MODEL}/{CORPUS}/{CORPUS}_words_plus_triplets_{MODEL}_{run}.csv'

        if os.path.exists(output_word_run_filepath):
            words_with_triplets_df = pd.read_csv(output_word_run_filepath)

        else:
            print(f'Aligning triplets from run {run} to words in corpus...')

            # read in triplets data
            with open(f'../data/output/{MODEL}/{CORPUS}/{MODEL}_triplets_{CORPUS}_{run}.json', 'r', encoding='utf-8') as f:
                triplets = json.load(f)

            # align triplets to words in word data
            triplet_map = compile_output_gpt(triplets, words_df, CORPUS)
            words_with_triplets_df = add_triplets_to_words_df(triplet_map, words_df, CORPUS)
            words_with_triplets_df.to_csv(output_word_run_filepath, index=False)
            # words_with_triplets_df = pd.read_csv(output_run_filepath)

        print(f'Adding triplets from run {run} to eye movement data...')

        # add triplets to eye mov data
        if CORPUS == 'onestop':
            df = pd.merge(eye_df, words_with_triplets_df[['article_title', 'difficulty_level', 'paragraph_id', 'ianum', 'triplets', 'n_triplets']],
                              how='left', on=['article_title', 'difficulty_level', 'paragraph_id', 'ianum'])
            df['text_id'] = [f'{article_batch}-{article_id}-{difficulty_level}'
                                     for article_batch, article_id, difficulty_level in
                                     zip(eye_df['article_batch'].tolist(), eye_df['article_id'].tolist(), eye_df['difficulty_level'].tolist())]
        elif CORPUS in ['meco','provo']:
            words_with_triplets_df['text_id'] = words_with_triplets_df['text_id'].astype(str)
            words_with_triplets_df['ianum'] = words_with_triplets_df['ianum'].astype(str)
            eye_df['text_id'] = eye_df['text_id'].astype(str)
            eye_df['ianum'] = eye_df['ianum'].astype(str)
            df = pd.merge(eye_df, words_with_triplets_df[['text_id', 'ianum', 'triplets', 'n_triplets']],
                              how='left', on=['text_id', 'ianum'])
        else:
            raise NotImplementedError(f'Corpus {CORPUS} not implemented')

        df['run_id'] = [run for i in range(len(df))]
        # df.to_csv(output_eye_run_filepath, index=False)
        datasets.append(df)

# merge data of all runs
df = pd.concat(datasets, ignore_index=True)
df.to_csv(f'../data/output/{MODEL}/{CORPUS}/{CORPUS}_eye_mov_plus_triplets_{MODEL}.csv', index=False)
for run, group in df.groupby(['run_id']):
    print(run)
    print(group['text_id'].unique())