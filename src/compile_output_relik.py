# Authors: Konstantin and Adrielli
import pandas as pd
import os
import re
import json
from collections import defaultdict
from process_corpus import check_alignment
import numpy as np

def map_text_to_step_outputs(model_name, corpus_name, step_dir, threshold, window_size, text_filepath, level = 'text'):

    model_name = re.escape(model_name)

    if corpus_name in ['meco', 'provo']:
        text_ids = pd.read_csv(text_filepath)['text_id'].tolist()

    elif corpus_name == 'onestop':
        df = pd.read_csv(text_filepath)
        if level == 'article':
            text_ids = [f'{info[0]}-{info[1]}-{info[2]}'
                        for info, rows in
                        df.groupby(['article_batch', 'article_id', 'difficulty_level'])]
        else:
            text_ids = [f'{article_batch}-{article_id}-{paragraph_id}-{difficulty_level}'
                        for article_batch, article_id, paragraph_id, difficulty_level in
                        zip(df['article_batch'].tolist(), df['article_id'].tolist(), df['paragraph_id'].tolist(),
                            df['difficulty_level'].tolist())]
    else:
        raise NotImplementedError(f'Corpus name {corpus_name} not implemented. Choose from onestop, meco or provo.')

    step_filepaths_per_text = defaultdict(list)
    for text_id in text_ids:
        if level == 'article':
            text_id = text_id.replace('-Adv', '-(\d+)-Adv').replace('-Ele', '-(\d+)-Ele')
        pattern = rf"output_step_(\d+)_{model_name}_{corpus_name}_{text_id}_{threshold}_{window_size}.json"
        filename_re = re.compile(pattern, re.IGNORECASE)
        for file_name in os.listdir(step_dir):
            match = filename_re.match(file_name)
            if not match:
                continue
            step_num = int(match.group(1))
            step_filepaths_per_text[text_id].append((step_num, file_name))
        step_filepaths_per_text[text_id] = sorted(step_filepaths_per_text[text_id])
        # print(f'{text_id}: {step_filepaths_per_text[text_id]}')

    return step_filepaths_per_text

def compile_steps_relik(step_dir, dir_to_save_triplets, model_name, corpus_name, threshold, window_size, text_filepath, level='text'):

    all_full, all_drops, all_adds = [], [], []

    if not os.path.isdir(dir_to_save_triplets): os.mkdir(dir_to_save_triplets)

    files_by_type = map_text_to_step_outputs(model_name, corpus_name, step_dir, threshold, window_size, text_filepath, level)
    counter = 0

    for text_type, files in files_by_type.items():

        if corpus_name == 'onestop':
            text_type = text_type.split('-')

        rows_all, rows_drop, rows_add = [],[],[]

        files.sort(key=lambda x: x[0])
        prev_set = set()
        seen_ever = set()

        for step, file_name in files:

            filepath = step_dir + "/" + file_name
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            simplified_triplets = []
            triplet_scores = []
            for raw in data.get("triplets", []):
                # raw is [head, rel, tail, score]
                head, rel, tail, *rest = raw
                score = rest[0] if rest else None

                # HEAD
                if isinstance(head, list):
                    canon_h = head[2] if len(head) > 2 else ""
                    surface_h = head[3] if len(head) > 3 else ""
                    head_name = f"{canon_h} | {surface_h}" if surface_h and surface_h != canon_h else canon_h
                else:
                    head_name = str(head)

                # TAIL
                if isinstance(tail, list):
                    canon_t = tail[2] if len(tail) > 2 else ""
                    surface_t = tail[3] if len(tail) > 3 else ""
                    tail_name = f"{canon_t} | {surface_t}" if surface_t and surface_t != canon_t else canon_t
                else:
                    tail_name = str(tail)

                simplified_triplets.append((head_name, rel, tail_name))
                triplet_scores.append(score)
            # print("Simplified triplets:", simplified_triplets)

            curr_set = set(simplified_triplets)
            impacted = curr_set != prev_set
            added_trips = [t for t in curr_set if t not in prev_set]
            dropped_trips = [t for t in prev_set if t not in curr_set]
            new_trips = [t for t in added_trips if t not in seen_ever]

            # find number of triplets in which current word occurs
            n_triplets_activated = 0
            mentions = []
            if curr_set:
                for t in curr_set:
                    for element in [t[0],t[-1]]:
                        if '|' in element: mentions.append(element.split(' | ')[-1])
                        else:
                            mentions.append(element)
                if mentions:
                    for mention in mentions:
                        if data["text"].split()[-1] == mention:
                            n_triplets_activated += 1

            n_new_triplets_activated = 0
            if new_trips:
                mentions = []
                for t in new_trips:
                    for element in [t[0], t[-1]]:
                        if '|' in element:
                            mentions.append(element.split(' | ')[-1])
                        else:
                            mentions.append(element)
                if mentions:
                    for mention in mentions:
                        if data["text"].split()[-1] == mention:
                            n_new_triplets_activated += 1

            if corpus_name in ['meco','provo']:

                rows_all.append({
                    "text_id": counter,
                    "text_type": text_type,
                    "output_step": step - 1,
                    "current_word": data["text"].split()[-1],
                    "triplets_impacted": 1 if impacted else 0,
                    "triplets_added": 1 if len(added_trips)>0 else 0,
                    "triplets_dropped": 1 if len(dropped_trips)>0 else 0,
                    "current_text": data["text"],
                    "total_triplets": simplified_triplets,
                    "triplet_scores": triplet_scores,
                    "n_triplets": len(simplified_triplets),
                    "n_triplets_added": len(added_trips),
                    "n_triplets_dropped": len(dropped_trips),
                    "n_triplets_new": len(new_trips),
                    "n_triplets_activated": n_triplets_activated,
                    "n_new_triplets_activated": n_new_triplets_activated
                })

                if new_trips:
                    seen_ever.update(new_trips)
                    rows_add.append({
                        "text_id": counter,
                        "text_type": text_type,
                        "output_step": step - 1,
                        "current_word": data["text"].split()[-1],
                        "current_text": data["text"],
                        "new_triplets": new_trips,
                        "total_triplets": simplified_triplets,
                        "triplet_scores": triplet_scores,
                    })

                if dropped_trips:
                    rows_drop.append({
                        "text_id": counter,
                        "text_type": text_type,
                        "output_step": step - 1,
                        "current_word": data["text"].split()[-1],
                        "current_text": data["text"],
                        "dropped_triplets": dropped_trips,
                        "total_triplets": simplified_triplets,
                        "triplet_scores": triplet_scores,
                    })

            elif corpus_name == 'onestop':

                if level == 'article':
                    pattern = re.compile(r'(\d+)(?=-(Ele|Adv))')
                    match = pattern.search(file_name)
                    if match: text_type[2] = match.group(1)

                rows_all.append({
                    "article_batch": text_type[0],
                    'article_id': text_type[1],
                    'paragraph_id': text_type[2],
                    'difficulty_level': text_type[3],
                    "output_step": step - 1,
                    "current_word": data["text"].split()[-1],
                    "triplets_impacted": 1 if impacted else 0,
                    "triplets_added": 1 if len(added_trips) > 0 else 0,
                    "triplets_dropped": 1 if len(dropped_trips) > 0 else 0,
                    "current_text": data["text"],
                    "total_triplets": simplified_triplets,
                    "triplet_scores": triplet_scores,
                    "n_triplets": len(simplified_triplets),
                    "n_triplets_added": len(added_trips),
                    "n_triplets_dropped": len(dropped_trips),
                    "n_triplets_new": len(new_trips),
                    "n_triplets_activated": n_triplets_activated,
                    "n_new_triplets_activated": n_new_triplets_activated
                })

                if new_trips:
                    seen_ever.update(new_trips)
                    rows_add.append({
                        "article_batch": text_type[0],
                        'article_id': text_type[1],
                        'paragraph_id': text_type[2],
                        'difficulty_level': text_type[3],
                        "output_step": step - 1,
                        "current_word": data["text"].split()[-1],
                        "current_text": data["text"],
                        "new_triplets": new_trips,
                        "total_triplets": simplified_triplets,
                        "triplet_scores": triplet_scores,
                    })

                if dropped_trips:
                    rows_drop.append({
                        "article_batch": text_type[0],
                        'article_id': text_type[1],
                        'paragraph_id': text_type[2],
                        'difficulty_level': text_type[3],
                        "output_step": step - 1,
                        "current_word": data["text"].split()[-1],
                        "current_text": data["text"],
                        "dropped_triplets": dropped_trips,
                        "total_triplets": simplified_triplets,
                        "triplet_scores": triplet_scores,
                    })
            else:
                raise NotImplementedError(f'Corpus {corpus_name} not implemented. Choose meco, provo, or onestop.')

            prev_set = curr_set

        counter += 1

        if corpus_name == 'onestop' and level == 'article':
            text_type.pop(2)

        if len(rows_all) > 0:
            df_full = (pd.DataFrame(rows_all).sort_values(["output_step"]).reset_index(drop=True))
            df_full.to_csv(f"{dir_to_save_triplets}/full_{model_name}_{corpus_name}_{text_type}.csv",
                           index=False)
            all_full.append(df_full)
        if len(rows_add) > 0:
            additions_df = pd.DataFrame(rows_add).sort_values(["output_step"]).reset_index(drop=True)
            additions_df.to_csv(f"{dir_to_save_triplets}/additions_{model_name}_{corpus_name}_{text_type}.csv",
                                index=False)
            all_adds.append(additions_df)
        if len(rows_drop) > 0:
            deletions_df = pd.DataFrame(rows_drop).sort_values(["output_step"]).reset_index(drop=True)
            deletions_df.to_csv(f"{dir_to_save_triplets}/deletions_{model_name}_{corpus_name}_{text_type}.csv",
                                index=False)
            all_drops.append(deletions_df)

    if corpus_name in ['meco', 'provo']:
        columns_sort = ["text_id", "output_step"]
    elif corpus_name == 'onestop':
        columns_sort = ["article_batch", "article_id", "difficulty_level", "paragraph_id", "output_step"]
    else:
        raise NotImplementedError(f'Corpus {corpus_name} not implemented.')

    if len(all_full) > 0:
        all_full_df = pd.concat(all_full, ignore_index=True)
        all_full_df = all_full_df.sort_values(columns_sort).reset_index(drop=True)
        all_full_df.to_csv(f"{dir_to_save_triplets}/full_{model_name}_{corpus_name}.csv", index=False)
    else:
        raise ValueError(f'No data found.')

    if len(all_adds) > 0:
        all_adds_df = pd.concat(all_adds, ignore_index=True)
        all_adds_df = all_adds_df.sort_values(columns_sort).reset_index(drop=True)
        all_adds_df.to_csv(f"{dir_to_save_triplets}/additions_{model_name}_{corpus_name}.csv", index=False)
    if len(all_drops) > 0:
        all_drops_df = pd.concat(all_drops, ignore_index=True)
        all_drops_df = all_drops_df.sort_values(columns_sort).reset_index(drop=True)
        all_drops_df.to_csv(f"{dir_to_save_triplets}/deletions_{model_name}_{corpus_name}.csv", index=False)



def add_triplets_to_eye_data(corpus_name, eye_df, triplets_df, level='text'):


    if corpus_name in ['meco','provo']:

        triplets_df.rename(columns={"output_step": "ianum"}, inplace=True)
        triplets_df.rename(columns={"current_word": "ia"}, inplace=True)
        df = pd.merge(eye_df, triplets_df[['text_id', 'ianum', 'ia', 'total_triplets', 'triplet_scores',
                                           "triplets_impacted", "triplets_added", "triplets_dropped", "n_triplets",
                                           "n_triplets_added", "n_triplets_dropped", "n_triplets_new",
                                           "n_triplets_activated", "n_new_triplets_activated"]], how='left',
                      on=['text_id', 'ianum', 'ia'])

    elif corpus_name == 'onestop':

        if level == 'article':
            triplets_df.rename(columns={"output_step": "article_ianum"}, inplace=True)
            triplets_df.rename(columns={"current_word": "ia"}, inplace=True)
            df = pd.merge(eye_df,
                          triplets_df[['article_batch', 'article_id', 'difficulty_level', 'paragraph_id', 'article_ianum',
                                       'ia', 'total_triplets', 'triplet_scores', "triplets_impacted",
                                       "triplets_added", "triplets_dropped", "n_triplets", "n_triplets_added",
                                       "n_triplets_dropped", "n_triplets_new", "n_triplets_activated", "n_new_triplets_activated"]], how='left',
                          on=['article_batch', 'article_id', 'difficulty_level', 'paragraph_id', 'article_ianum', 'ia'])
        else:
            triplets_df.rename(columns={"output_step": "ianum"}, inplace=True)
            triplets_df.rename(columns={"current_word": "ia"}, inplace=True)
            df = pd.merge(eye_df,
                          triplets_df[['article_batch', 'article_id', 'difficulty_level', 'paragraph_id', 'ianum',
                                       'ia', 'total_triplets', 'triplet_scores', "triplets_impacted",
                                       "triplets_added", "triplets_dropped", "n_triplets", "n_triplets_added",
                                       "n_triplets_dropped", "n_triplets_new", "n_triplets_activated", "n_new_triplets_activated"]], how='left',
                          on=['article_batch', 'article_id', 'difficulty_level', 'paragraph_id', 'ianum', 'ia'])

        df['text_id'] = [f'{article_batch}-{article_id}-{difficulty_level}'
                         for article_batch, article_id, difficulty_level in
                         zip(df['article_batch'].tolist(), df['article_id'].tolist(), df['difficulty_level'].tolist())]

    else:
        raise NotImplementedError(f'Corpus {corpus_name} not implemented. Choose between "meco", "onestop"')

    # add summed scores
    # df['triplet_scores'] = df['triplet_scores'].apply(lambda x: x.replace('[', '').replace(']', '').split(', '))
    # df['sum_scores'] = [sum([float(score.strip()) for score in triplets]) if any(triplets) else 0 for triplets in df['triplet_scores'].tolist()]

    return df

def main():

    corpus_name = 'provo' # onestop meco
    model_name = 'relik-cie-xl'
    eye_filepath = f'../data/processed/{corpus_name}_eye_mov.csv'
    step_dir = f'../data/output/step_outputs_{corpus_name}_{model_name}/_2025_09_08_17-59-36' # ../data/output/step_outputs_onestop/_2025_08_31_02-00-33 (article level) or _2025_08_23_10-52-11 (paragraph level) ../data/output/step_outputs_meco/_2025_07_31_14-54-48
    threshold = '0.1'
    window_size = '128' # None
    text_filepath = f'../data/processed/{corpus_name}_texts.csv'
    dir_to_save_triplets = f'../data/output/all_outputs_{corpus_name}_{model_name}' # ../data/output/all_outputs_{corpus_name}/article_level
    dir_to_save_final_data = f'../data/output/{corpus_name}_eye_mov_plus_triplets_{model_name}.csv' # ../data/output/eye_data_plus_triplets_{corpus_name}_article.csv
    level = 'text' # 'article'

    # read in eye mov data
    eye_df = pd.read_csv(eye_filepath)

    # compile all texts
    # compile_steps_relik(step_dir, dir_to_save_triplets, model_name, corpus_name, threshold, window_size, text_filepath, level)
    triplets_df = pd.read_csv(f"{dir_to_save_triplets}/full_{model_name}_{corpus_name}.csv")
    # triplets_df.to_csv(f"{dir_to_save_triplets}/full_{model_name}_{corpus_name}1.csv", index=False)

    # check alignment between eye mov data and triplet data
    check_alignment(corpus_name, triplets_df, eye_df, level)

    # add total_triplets and n_triplets to eye data
    final_df = add_triplets_to_eye_data(corpus_name, eye_df, triplets_df, level)
    final_df.to_csv(dir_to_save_final_data)

if __name__ == '__main__':
    main()