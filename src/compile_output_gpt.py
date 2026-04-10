import string
import pandas as pd
import json
import re
import os

def add_sent_id(words_df, corpus):

    if corpus in ['meco','provo']:
        if 'sent_id' not in words_df.columns:
            words_df = (words_df.groupby(['text_id'])
                        .apply(lambda group: assign_sentence(group)).reset_index(drop=True))
            words_df.to_csv(f'../data/processed/{corpus}_words.csv', index=False)

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

def assign_article_level_ids(group:pd.DataFrame):

    paragraphs = group['paragraph'].unique().tolist()
    article = ' '.join(paragraphs)
    sentences = re.split(r'\.\s|\."\s|\.”\s|\?\s|!\s|\?”\s', article)
    words = group['ia'].tolist()

    word_to_sentence = dict()
    word_index = 0
    for sent_id, sentence in enumerate(sentences):
        start, end = (word_index, word_index + len(sentence.split()))
        while (word_index < len(words) and
               start <= word_index < end):
            word_to_sentence[word_index] = sent_id
            word_index += 1

    group = group.copy()
    group['article_ianum'] = [i for i in range(len(group))]
    group['article_sent_id'] = group['article_ianum'].map(word_to_sentence)
    # for word, sent_id in zip(words_df['ia'].tolist(), words_df['article_sent_id'].tolist()):
    #     print(word, sent_id)

    return group

def find_span_indices(span:str, words:pd.DataFrame):
    """
    Find the start and end indices of a span in a list of words.
    :param span: mention in triplet
    :param words: words in eye movement data
    :return: indices of the span in the word list
    """

    span_chars = set(span.lower())
    # remove punctuation
    span_chars = {c for c in span_chars if c.isalnum()}
    span_len = len(span.split())
    # print(span, span_chars, span_len)

    # create word list
    word_list = words['ia'].tolist()

    best_match = None
    best_score = -1
    best_candidate = ''
    # start search from end of word list (most recent words are preferred)
    for start in reversed(range(len(word_list))):
        for end in reversed(range(start + 1, len(word_list) + 1)):
            # skip candidates with length too different
            candidate_len = end - start
            if abs(candidate_len - span_len) > 3:
                continue
            candidate = ' '.join(word_list[start:end])
            candidate_chars = set(candidate.lower())
            # remove punctuation
            candidate_chars = {c for c in candidate_chars if c.isalnum()}
            # print(start, end, candidate, candidate_chars)
            if len(candidate_chars) > 0:
                score = len((span_chars & candidate_chars))/((len(candidate_chars) + len(span_chars))/2)
            # skip candidates with only punctuation(s) (therefore after punct removal, nothing is left)
            else:
                continue
                # raise ValueError(candidate, candidate_chars, span, span_chars)
            # print(score)
            if score > best_score:
                best_score = score
                best_match = [(start, end)]
                best_candidate = candidate
            elif score == best_score:
                # prefer candidate with the closest length to the mention span
                if abs(candidate_len - span_len) < abs(len(best_candidate.split()) - span_len):
                    best_match = [(start, end)]
                    best_candidate = candidate
                elif abs(candidate_len - span_len) == abs(len(best_candidate.split()) - span_len):
                    # prefer the candidate whose words are the most similar to the mention span (if length difference is the same)
                    current_words = [word.strip(string.punctuation) for word in candidate.lower().split()]
                    span_words = [word.strip(string.punctuation) for word in span.lower().split()]
                    best_candidate_words = [word.strip(string.punctuation) for word in best_candidate.lower().split()]
                    # how many words are not matching between candidate and span, considering word order + length difference
                    diff_words_current = sum([1 for candidate_word, span_word in zip(current_words, span_words) if candidate_word != span_word]) + abs(len(current_words) - len(span_words))
                    # how many words are not matching between the best candidate and span, considering word order + length difference
                    diff_words_best = sum([1 for best_word, span_word in zip(best_candidate_words, span_words) if best_word != span_word]) + abs(len(best_candidate_words) - len(span_words))
                    if diff_words_current < diff_words_best:
                        best_match = [(start, end)]
                        best_candidate = candidate
                    # if they are exactly the same (e.g. "it", "she", "Google"), add candidate to list of best
                    elif diff_words_current == diff_words_best:
                        best_match.append((start, end))

    # print(best_match, best_candidate, best_score)

    return best_match

def align_triplets_to_words(text_triplets:list[dict], words_df:pd.DataFrame, text_type=''):
    """
    Given a list of triplets and a list of words, align the triplets to the words based on the triplet mentions.
    :return: dict with word ids as keys and dict with word form and triplets as values
    """
    # does not account for re-analysis (e.g. new triplets extracted from sentence n-1 do not get added to words in sentence n-1 nor do they replace the triplets in n-1)

    word_to_triplet = dict()
    open_pos_tags = {'ADJ','ADV','INTJ','NOUN','PROPN','VERB'}

    if text_type == '_articles':
        ianum_col = 'article_ianum'
        sent_ids_col = 'article_sent_id'
    else:
        ianum_col = 'ianum'
        sent_ids_col = 'sent_id'

    # create dictionary mapping word ids to triplets
    for word, word_id in zip(words_df['ia'].tolist(), words_df[ianum_col].tolist()):
        word_to_triplet[str(word_id)] = {'word': word, 'triplets': [], 'new_triplets':[], 'distance_to_first_mention':[]}

    # make sure words and triplets were extracted from the same sentence
    for step_index, sentence_triplets in enumerate(text_triplets):
        # print(f'Processing step: {sentence_triplets["step"]}/{len(text_triplets)}')
        # print(f'Context: {sentence_triplets["context"]}')
        # match triplet to words in current and previous sentences
        words_df_filtered = words_df[words_df[sent_ids_col] <= sentence_triplets['step']]

        # find word indices of mentions in triplet
        for triplet in sentence_triplets['triplets']:
            mention_indices = []
            # only align triplet if triplet not already assigned to words in previous steps (repeated triplets across steps)
            if triplet in text_triplets[step_index-1]['triplets'] and step_index > 0:
                continue
            else:
                # print(f'Aligning triplet: {triplet}')
                words = words_df_filtered['ia'].tolist()
                pos_tags = words_df_filtered['pos_tag'].tolist()
                for key, value in triplet.items():
                    # entity_1, relation, entity_2
                    if key != 'confidence':
                        # for each mention, find the indices in the word list
                        indices = find_span_indices(value['mention'], words_df_filtered)
                        if not indices:
                            # if corpus == 'onestop':
                            #     print(words_df['article_batch'].tolist()[0], words_df['article_id'].tolist()[0], words_df['difficulty_level'].tolist()[0], words_df['paragraph_id'].tolist()[0])
                            # elif corpus in ['meco','provo']:
                            #     print(words_df['text_id'].tolist()[0])
                            # for word, sent_id in zip(words_df['ia'],words_df['sent_id']):
                            #     print(word, sent_id)
                            raise ValueError(f"Mention '{value['mention']}' in triplet {triplet} and step {sentence_triplets['step']} not in word list: {words}")
                        mention_indices.append(indices)
                # print(mention_indices)

                # disambiguate mentions that had more than one best candidate by giving preference to candidate in the same sentence as the other mentions in triplet.
                for i_c, candidate_indices in enumerate(mention_indices):
                    if len(candidate_indices) > 1:
                        # print(candidate_indices)
                        scores = []
                        other_mention_sent_id_1 = words_df_filtered.sent_id.values[mention_indices[i_c-1][0][0]:mention_indices[i_c-1][0][1]][0]
                        # print(words_df_filtered.ia.values[mention_indices[i_c-1][0][0]:mention_indices[i_c-1][0][1]])
                        # print(other_mention_sent_id_1)
                        other_mention_sent_id_2 = words_df_filtered.sent_id.values[mention_indices[i_c-2][0][0]:mention_indices[i_c-2][0][1]][0]
                        # print(words_df_filtered.ia.values[mention_indices[i_c - 2][0][0]:mention_indices[i_c - 2][0][1]])
                        # print(other_mention_sent_id_2)
                        for candidate in candidate_indices:
                            # print(candidate)
                            score = 0
                            mention_candidate_sent_id = words_df_filtered.sent_id.values[candidate[0]:candidate[1]][0]
                            # print(words_df_filtered.ia.values[candidate[0]:candidate[1]])
                            # print(mention_candidate_sent_id)
                            if mention_candidate_sent_id == other_mention_sent_id_1:
                                score += 1
                            if mention_candidate_sent_id == other_mention_sent_id_2:
                                score += 1
                            scores.append(score)
                        # if the other mentions are not in the same sentence as any of the candidates, prefer most recent candidate (first index).
                        mention_indices[i_c] = [mention_indices[i_c][scores.index(max(scores))]]
                        # print(mention_indices[i_c])

                # find which mention is the latest in the text
                max_index, min_index = 0, len(words_df['ia'].tolist())-1
                latest_mention_index = None
                earliest_mention_index = None
                distance_between_mentions = None
                for mention_index in mention_indices:
                    # print(mention_index)
                    # print(words_df_filtered.ia.values[mention_index[0][0]:mention_index[0][1]])
                    if mention_index:
                        if mention_index[0][-1]-1 >= max_index:
                            max_index = mention_index[0][-1]-1
                            latest_mention_index = mention_index[0]
                        if mention_index[0][-1]-1 <= min_index:
                            min_index = mention_index[0][-1]-1
                            earliest_mention_index = mention_index[0]
                if latest_mention_index and earliest_mention_index:
                    # compute distance between lastest and earliest mention in text
                    distance_between_mentions = latest_mention_index[-1]-1 - earliest_mention_index[0]
                # print(earliest_mention_index, latest_mention_index)
                # print(distance_between_mentions)

                # find which word in mention span gets the triplet
                if latest_mention_index:
                    # find last content word
                    mention_words = words[latest_mention_index[0]:latest_mention_index[-1]]
                    mention_pos_tags = pos_tags[latest_mention_index[0]:latest_mention_index[-1]]
                    # print('Latest mention index:', latest_mention_index)
                    # print('Latest mention word:', words[latest_mention_index[0]:latest_mention_index[-1]])
                    content_words = [word for word, pos_tag in zip(mention_words, mention_pos_tags) if pos_tag in open_pos_tags]
                    if content_words:
                        last_content_word = content_words[-1]
                        # print('Latest content word: ', last_content_word)
                        last_content_word_id = len(words) - 1 - words[::-1].index(last_content_word)
                        # print('triplet gets added at this word: ', last_content_word, last_content_word_id)
                        # add triplet only if last content word of last mention is in the last sentence of current step
                        if words_df_filtered[sent_ids_col].tolist()[last_content_word_id] == sentence_triplets['step']:
                            # add triplet to all following words
                            for word_id in word_to_triplet.keys():
                                if int(word_id) >= int(last_content_word_id):
                                    if triplet not in word_to_triplet[str(word_id)]['triplets']:
                                        word_to_triplet[str(word_id)]['triplets'].append(triplet)
                            # add triplet only to current word (new triplet):
                            if str(last_content_word_id) in word_to_triplet.keys():
                                if triplet not in word_to_triplet[str(last_content_word_id)]['new_triplets']:
                                    word_to_triplet[str(last_content_word_id)]['new_triplets'].append(triplet)
                                    word_to_triplet[str(last_content_word_id)]['distance_to_first_mention'].append(distance_between_mentions)

    return word_to_triplet

def compile_output_gpt(triplets:dict, words_df:pd.DataFrame, corpus='', text_type=''):
    """
    :return: dictionary mapping word ids to triplets
    """

    triplet_map = dict()

    if corpus == 'onestop':
        if text_type == '_articles':
            for text_metadata in triplets:
                text_triplets = text_metadata['extracted_triplets']
                article_batch = text_metadata['article_batch']
                article_id = text_metadata['article_id']
                difficulty_level = text_metadata['difficulty_level']
                # if article_batch == 1 and article_id == 0 and difficulty_level == 'Adv':
                # filter words data for the current text
                words_df_filtered = words_df[(words_df['article_batch'] == article_batch) &
                                            (words_df['article_id'] == article_id) &
                                            (words_df['difficulty_level'] == difficulty_level)]
                print('Processing article:', article_batch, article_id, difficulty_level)
                word_to_triplet = align_triplets_to_words(text_triplets, words_df_filtered, text_type)
                triplet_map[f'{article_batch}-{article_id}-{difficulty_level}'] = word_to_triplet
        else:
            # onestop paragraph level
            for text_metadata in triplets:
                text_triplets = text_metadata['extracted_triplets']
                article_batch = text_metadata['article_batch']
                article_id = text_metadata['article_id']
                difficulty_level = text_metadata['difficulty_level']
                paragraph_id = text_metadata['paragraph_id']
                # if article_batch == 1 and article_id == 0 and difficulty_level == 'Adv' and paragraph_id == 1:
                # filter words data for the current text
                words_df_filtered = words_df[(words_df['article_batch'] == article_batch) &
                                           (words_df['article_id'] == article_id) &
                                           (words_df['difficulty_level'] == difficulty_level) &
                                           (words_df['paragraph_id'] == paragraph_id)]
                # print('Processing paragraph:', article_batch, article_id, difficulty_level, paragraph_id)
                word_to_triplet = align_triplets_to_words(text_triplets, words_df_filtered)
                triplet_map[f'{article_batch}-{article_id}-{difficulty_level}-{paragraph_id}'] = word_to_triplet

    elif corpus in ['meco','provo']:
        words_df['text_id'] = words_df['text_id'].astype(str)
        for text_metadata in triplets:
            text_triplets = text_metadata['extracted_triplets']
            text_id = str(text_metadata['text_id'])
            # filter words data for the current text
            words_df_filtered = words_df[(words_df['text_id'] == text_id)]
            word_to_triplet = align_triplets_to_words(text_triplets, words_df_filtered)
            triplet_map[text_id] = word_to_triplet
    else:
        raise NotImplementedError(f'Corpus {corpus} not implemented')

    return triplet_map

def add_triplets_to_words_df(triplet_map:dict, words_df:pd.DataFrame, corpus:str, text_type:str=''):

    all_rows = []

    if triplet_map:

        ianum_col = 'ianum'
        if text_type == '_articles':
            ianum_col = 'article_ianum'

        for i, row in words_df.iterrows():

            if corpus in ['meco','provo']:
                key= str(row['text_id'])
            elif corpus=='onestop':
                if text_type == '_articles':
                    key = f"{row['article_batch']}-{row['article_id']}-{row['difficulty_level']}"
                else:
                    key = f"{row['article_batch']}-{row['article_id']}-{row['difficulty_level']}-{row['paragraph_id']}"
            else:
                raise NotImplementedError(f'Corpus {corpus} not implemented')

            word_id = str(row[ianum_col])

            if key in triplet_map.keys():
                triplets = triplet_map.get(key, {}).get(word_id, {}).get('triplets', [])
                new_triplets = triplet_map.get(key, {}).get(word_id, {}).get('new_triplets', [])
                distances = triplet_map.get(key, {}).get(word_id, {}).get('distance_to_first_mention', [])
                new_row = row.to_dict()
                new_row['triplets'] = triplets
                new_row['n_triplets'] = len(triplets)
                new_row['new_triplets'] = new_triplets
                new_row['n_new_triplets'] = len(new_triplets)
                new_row['triplet_added'] = 1 if len(new_triplets) > 0 else 0
                new_row['distance_to_first_mention'] = distances
                if distances:
                    new_row['agg_distance'] = sum(distances)/len(distances)
                else:
                    new_row['agg_distance'] = 0
                all_rows.append(new_row)

        return pd.DataFrame(all_rows)

    else:
        raise ValueError(f'Triplet mapping is empty.')


def merge_triplets_and_words(triplets_run_filepath, output_word_run_filepath, words_df, corpus, text_type, run):

    # read in triplets data
    with open(triplets_run_filepath, 'r', encoding='utf-8') as f:
        triplets = json.load(f)

    # align triplets to words in word data
    if corpus == 'onestop':
        words_df = (words_df.groupby(['article_batch', 'article_id', 'difficulty_level'])
                    .apply(lambda group: assign_article_level_ids(group), include_groups=True).reset_index(drop=True))

    triplet_map = compile_output_gpt(triplets, words_df, corpus, text_type=text_type)
    words_with_triplets_df = add_triplets_to_words_df(triplet_map, words_df, corpus, text_type=text_type)
    words_with_triplets_df.to_csv(output_word_run_filepath, index=False)
    # words_with_triplets_df = pd.read_csv(output_word_filepath)
    words_with_triplets_df['run_id'] = [run for i in range(len(words_with_triplets_df))]

    return words_with_triplets_df

def merge_triplets_and_eye(corpus, eye_df, words_with_triplets_df, output_eye_run_filepath, text_type, run):

    if corpus == 'onestop':
        if text_type == '_articles':
            eye_with_triplets_df = pd.merge(eye_df, words_with_triplets_df[
                ['article_batch', 'article_id', 'difficulty_level', 'article_ianum', 'triplets', 'n_triplets',
                 'new_triplets', 'n_new_triplets', 'triplet_added', 'distance_to_first_mention', 'agg_distance']],
                          how='left', on=['article_batch', 'article_id', 'difficulty_level', 'article_ianum'])
        else:
            eye_with_triplets_df = pd.merge(eye_df, words_with_triplets_df[
                ['article_batch', 'article_id', 'difficulty_level', 'paragraph_id', 'ianum', 'triplets', 'n_triplets',
                 'new_triplets', 'n_new_triplets', 'triplet_added', 'distance_to_first_mention', 'agg_distance']],
                          how='left', on=['article_batch', 'article_id', 'difficulty_level', 'paragraph_id', 'ianum'])

    elif corpus in ['meco', 'provo']:
        words_with_triplets_df['text_id'] = words_with_triplets_df['text_id'].astype(str)
        words_with_triplets_df['ianum'] = words_with_triplets_df['ianum'].astype(str)
        eye_df['text_id'] = eye_df['text_id'].astype(str)
        eye_df['ianum'] = eye_df['ianum'].astype(str)
        eye_with_triplets_df = pd.merge(eye_df, words_with_triplets_df[
            ['text_id', 'ianum', 'triplets', 'n_triplets', 'new_triplets',
             'n_new_triplets', 'triplet_added', 'distance_to_first_mention', 'agg_distance']],
                      how='left', on=['text_id', 'ianum'])

    else:
        raise NotImplementedError(f'Corpus {corpus} not implemented')

    eye_with_triplets_df['run_id'] = [run for i in range(len(eye_with_triplets_df))]
    eye_with_triplets_df.to_csv(output_eye_run_filepath, index=False)

    return eye_with_triplets_df

def align_triplets(words_df, eye_df, corpus, triplets_run_filepath, output_word_run_filepath, output_eye_run_filepath, text_type, run):

    print(f'Aligning triplets from run {run} to words in corpus...')

    words_with_triplets_df = merge_triplets_and_words(triplets_run_filepath, output_word_run_filepath, words_df, corpus, text_type, run)

    print(f'Adding triplets from run {run} to eye movement data...')

    eye_with_triplets_df = merge_triplets_and_eye(corpus, eye_df, words_with_triplets_df, output_eye_run_filepath, text_type, run)

    return words_with_triplets_df, eye_with_triplets_df

def main():

    corpus = 'onestop'
    model = 'gpt-4o-mini'
    n_runs = 1 # 10
    skip_if_path_exists = False
    merge_runs = False
    text_type = '' # '_articles' if article onestop, else ''

    # read eye movement word data
    words_df = pd.read_csv(f'../data/processed/{corpus}_words.csv')
    eye_df = pd.read_csv(f'../data/processed/{corpus}_eye_mov.csv')

    # generate sentence ids if not in dataframe
    add_sent_id(words_df, corpus)

    # iterate over runs
    datasets = []
    word_datasets = []
    for run in range(1, n_runs+1):

        output_eye_run_filepath = f'../data/output/{model}/{corpus}/{corpus}{text_type}_eye_mov_plus_triplets_{model}_{run}_dist.csv'
        output_word_run_filepath = f'../data/output/{model}/{corpus}/{corpus}{text_type}_words_plus_triplets_{model}_{run}_dist.csv'
        triplets_run_filepath = f'../data/output/{model}/{corpus}/{model}_triplets_{corpus}{text_type}_{run}.json'

        if skip_if_path_exists:

            if os.path.exists(output_eye_run_filepath):
                df = pd.read_csv(output_eye_run_filepath)
                datasets.append(df)
            else:
                if os.path.exists(output_word_run_filepath):
                    words_with_triplets_df = pd.read_csv(output_word_run_filepath)
                    if 'run_id' not in words_with_triplets_df.columns:
                        words_with_triplets_df['run_id'] = [run for i in range(len(words_with_triplets_df))]
                    word_datasets.append(words_with_triplets_df)
                else:
                    words_with_triplets_df, eye_with_triplets_df = align_triplets(words_df,
                                                                                 eye_df,
                                                                                 corpus,
                                                                                 triplets_run_filepath,
                                                                                 output_word_run_filepath,
                                                                                 output_eye_run_filepath,
                                                                                 text_type, run)
                    if merge_runs:
                        word_datasets.append(words_with_triplets_df)
                        datasets.append(eye_with_triplets_df)
        else:
            words_with_triplets_df, eye_with_triplets_df = align_triplets(words_df,
                                                                         eye_df,
                                                                         corpus,
                                                                         triplets_run_filepath,
                                                                         output_word_run_filepath,
                                                                         output_eye_run_filepath,
                                                                         text_type, run)
            if merge_runs:
                word_datasets.append(words_with_triplets_df)
                datasets.append(eye_with_triplets_df)


    if merge_runs:
        # merge data of all runs
        df = pd.concat(datasets, ignore_index=True)
        df.to_csv(f'../data/output/{model}/{corpus}/{corpus}{text_type}_eye_mov_plus_triplets_{model}.csv', index=False)
        df2 = pd.concat(word_datasets, ignore_index=True)
        df2.to_csv(f'../data/output/{model}/{corpus}/{corpus}{text_type}_words_plus_triplets_{model}.csv', index=False)

if __name__ == '__main__':
    main()