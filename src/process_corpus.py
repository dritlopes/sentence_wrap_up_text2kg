import pandas as pd
import spacy
import rdata
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os
import string

def find_keywords(texts):

    nlp = spacy.load("en_core_web_sm")
    content_pos = ['NOUN', 'PROPN']

    keywords = []
    for text in texts:
        doc = nlp(text)
        words = [token.text for token in doc if token.pos_ in content_pos]
        keyword = max(set(words), key=words.count)
        keywords.append(keyword)

    return keywords

def extract_provo_texts(data_dir):

    """
    Pre-process file with texts from Provo.
    :param data_dir: filepath to raw Provo text file.
    :return: Output dataframe with columns [text_id, text, keyword]
    """

    data = pd.read_csv(data_dir, encoding="ISO-8859-1")
    data = pd.DataFrame({'text': [text for text in data['Text'].unique()]})
    data["text"] = data["text"].str.replace("Ñ", "", regex=False)
    data["text"] = data["text"].str.replace("Õ", "", regex=False)
    data["text"] = data["text"].str.replace('"', '', regex=False)
    data['text_id'] = [i for i in range(len(data['text']))]
    data['keyword'] = find_keywords(data["text"].tolist())
    data = data[['text_id', 'keyword', 'text']]

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

    # add column with keyword (most frequent content word) for each text
    data['keyword'] = ['janus', 'shaka', 'doping', 'thylacine', 'wed', 'monocle', 'wine', 'orange', 'beekeeping', 'flag', 'nature', 'vehicle']
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
        raise NotImplementedError("Parameter `corpus_name` must be either `provo` or `meco`.")

    return dataset

def pre_process_provo_data(filepath):

    """
    Pre-process word-based data from provo.
    Returns: pre-processed data

    """

    df = pd.read_csv(filepath, encoding="ISO-8859-1")

    # select columns
    df = df[['Participant_ID', 'Text_ID', 'Word', 'Word_Number', 'IA_FIRST_FIXATION_DURATION',
             'IA_FIRST_RUN_DWELL_TIME', 'IA_DWELL_TIME', 'IA_SKIP', 'IA_REGRESSION_IN', 'IA_REGRESSION_OUT']]

    # drop nan values
    df.dropna(subset=['Text_ID', 'Word', 'Word_Number'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # starting indexing with at 0
    df['Text_ID'] = df['Text_ID'].apply(lambda x: int(x) - 1)
    df['Word_Number'] = df['Word_Number'].apply(lambda x: int(x) - 1)

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
                            'IA_SKIP': 'skip',
                            'IA_DWELL_TIME': 'total_dur',
                            'IA_FIRST_FIXATION_DURATION': 'first_fix_dur',
                            'IA_FIRST_RUN_DWELL_TIME': 'gaze_dur',
                            'IA_REGRESSION_IN': 'reg_in',
                            'IA_REGRESSION_OUT': 'reg_out',
                            'Participant_ID': 'participant_id'})
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

def pre_process_eye_data(corpus_name:str, filepath:str):

    if corpus_name == 'meco':
        eye_data = pre_process_meco_data(filepath)
    elif corpus_name == 'provo':
        eye_data = pre_process_provo_data(filepath)
    else:
        raise Exception(f'Corpus {corpus_name} not implemented.')

    return eye_data

def add_word_frequency(df, corpus_name, frequency_filepath):

    if corpus_name == 'meco':  # we use frequency file from meco corpus
        freq_col_name = 'zipf_freq'
        word_col_name = 'ia_clean'
        frequency_df = pd.read_csv(frequency_filepath, usecols=[freq_col_name, word_col_name])
        if 'lang' in frequency_df.columns:
            frequency_df = frequency_df[frequency_df['lang'] == 'english']
    elif corpus_name == 'Provo':  # we use SUBTLEX-UK
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

def check_alignment(words_df: pd.DataFrame, eye_df: pd.DataFrame):

    """
    Check alignment between word and fixation dataframes (whether word ids match).
    :param words_df: words dataframe
    :param eye_df: fixation dataframe
    """

    # for each word if and word in eye-movement dataframe, check if it's the same in word dataframe
    for id, data in eye_df.groupby(['participant_id', 'text_id']):
        text_words = words_df[words_df['text_id'] == id[1]]
        for eye_ia, eye_ianum in zip(data['ia'].tolist(), data['ianum'].tolist()):
            # find row in triplets df with word and word id from eye mov df
            assert not text_words[
                (text_words['output_step'] == eye_ianum) & (text_words['current_word'] == eye_ia)].empty, (
                print(f'Word {eye_ia} with word id {eye_ianum} in eye mov data not in triplet data. '
                      f'In triplet data, word id {eye_ianum} yields word "{text_words[text_words["output_step"] == eye_ianum]["current_word"].tolist()[0]}"'))

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

def add_word_surprisal(eye_df, corpus_name, text_filepath):

    path_to_save = text_filepath.replace('texts.csv', 'surprisal.csv')

    if os.path.exists(path_to_save):
        surprisal_df = pd.read_csv(path_to_save, sep='\t')

    else:
        print('Computing word surprisal...')

        # create data resource to add surprisal values where each row is a word
        if not os.path.exists(text_filepath):
            text_df = extract_texts(corpus_name, save_dir=text_filepath)
        else:
            text_df = pd.read_csv(text_filepath)
        trialids, words, word_ids = [], [], []
        for text_id, text in enumerate(text_df['text'].tolist()):
            text_words = text.split()
            words.extend(text_words)
            trialids.extend([text_id for i in range(len(text_words))])
            word_ids.extend([i for i in range(len(text_words))])
        surprisal_df = pd.DataFrame(data={'text_id': trialids,
                                          'ianum': word_ids,
                                          'ia': words})

        # check alignment with eye data
        check_alignment(surprisal_df, eye_df)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device ', str(device))

        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # compute surprisal values
        surprisal_values, model_tokens, corpus_tokens = calculate_surprisal_values(surprisal_df, model, tokenizer, device)
        surprisal_df['surprisal'] = surprisal_values

        # write out surprisal dataset
        directory = os.path.dirname(path_to_save)
        if not os.path.isdir(directory): os.mkdir(directory)
        surprisal_df.to_csv(path_to_save, sep='\t', index=False)

        # write out which words in the corpus are multi-tokens in the model
        path_to_save_multi_tokens = path_to_save.replace('.csv', '_multi_tokens.csv')
        with open(path_to_save_multi_tokens, 'w') as outfile:
            outfile.write(f'CORPUS_TOKEN\tMODEL_TOKEN\n')
            for model_token, corpus_token in zip(model_tokens, corpus_tokens):
                outfile.write(f'{corpus_token}\t{model_token}\n')

    return surprisal_df

def add_variables(variables:list[str], df:pd.DataFrame, corpus_name:str, text_filepath:str='', frequency_filepath:str='')->pd.DataFrame:

    """
    Add length, frequency and pos-tag to eye-tracking dataframe.
    Args:
        variables: list of possible variables (length, frequency, pos-tag).
        df: dataframe with eye-tracking data
        corpus_name: name of eye-tracking corpus
        frequency_filepath: filepath to frequency resource

    Returns: dataframe with eye-tracking data and variables added as columns.

    """

    if 'length' in variables:
        df['length'] = [len(str(word)) for word in df['ia'].tolist()]

    if 'frequency' in variables and frequency_filepath:
        df['frequency'] = add_word_frequency(df, corpus_name, frequency_filepath)

    if 'surprisal' in variables and text_filepath:
        surprisal_df = add_word_surprisal(df, corpus_name, text_filepath)
        df = pd.merge(df, surprisal_df[['text_id', 'ianum', 'surprisal']], how='left', on=['text_id', 'ianum'])

    if 'sent_length' in variables:
        if corpus_name == 'meco':
            sent_length = []
            for info, rows in df.groupby(['participant_id','text_id','sentnum']):
                len_sent = len(rows)
                sent_length.extend([len_sent for i in range(len(rows))])
            df['sent_length'] = sent_length

    if 'sent_mean_frequency' in variables and 'frequency' in df.columns:
        if corpus_name == 'meco':
            sent_mean_freq = []
            for info, rows in df.groupby(['participant_id','text_id','sentnum']):
                sent_mean_freq.extend([np.mean(rows['frequency'].tolist()) for i in range(len(rows))])
            df['sent_mean_frequency'] = sent_mean_freq

    if 'word_pos' in variables:
        if corpus_name == 'meco':
            abs_word_pos, norm_word_pos = [], []
            for info, rows in df.groupby(['participant_id','text_id','sentnum']):
                abs = [i for i in range(len(rows))]
                abs_word_pos.extend(abs)
                norm = (np.array(abs) - np.min(abs))/(np.max(abs) - np.min(abs))
                norm_word_pos.extend(norm)
            df['abs_word_pos'] = abs_word_pos
            df['norm_word_pos'] = norm_word_pos

    return df

def main():

    """
    Process corpus files.
    Returns: write out processed file:
    """

    # df = pd.read_csv('../data/raw/ia_Paragraph_ordinary.csv')

    # file with eye-tracking data
    eye_move_filepath = '../data/raw/joint_data_trimmed.csv'  # '../data/raw/Provo_Corpus-Eyetracking_Data.csv'
    # file with texts from corpus
    texts_filepath = '../data/processed/meco_texts.csv'
    # file with word frequency resource
    frequency_filepath = '../data/raw/wordlist_meco.csv'  # '../data/raw/Provo/SUBTLEX_UK.txt'
    # corpus name
    corpus_name = 'meco'  # 'Provo'
    # filepath to save out pre-processed eye-tracking data
    processed_eye_move_filepath = f'../data/processed/{corpus_name}_eye_mov.csv'

    print('Processing dataframe with eye movements...')
    eye_data = pre_process_eye_data(corpus_name, eye_move_filepath)
    # eye_data = pd.read_csv(processed_eye_move_filepath)
    eye_data = add_variables(
        ['length', 'frequency', 'surprisal', 'sent_length', 'sent_mean_frequency', 'word_pos'], # 'length', 'frequency', 'surprisal', 'sent_length', 'sent_mean_frequency', 'word_pos'
        eye_data, corpus_name, texts_filepath, frequency_filepath)
    eye_data.to_csv(processed_eye_move_filepath, index=False)

if __name__ == '__main__':
    main()