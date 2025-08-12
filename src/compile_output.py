# Authors: Konstantin and Adrielli
import pandas as pd
import os
import re
import json
from collections import defaultdict
from process_corpus import check_alignment
import numpy as np

def map_text_to_step_outputs(model_name, corpus_name, step_dir, threshold, window_size, text_filepath):

    model_name = re.escape(model_name)

    keywords = pd.read_csv(text_filepath)['keyword'].tolist()

    step_filepaths_per_text = defaultdict(list)

    for keyword in keywords:

        pattern = rf"output_step_(\d+)_{model_name}_{corpus_name}_{keyword}_{threshold}_{window_size}.json"
        filename_re = re.compile(pattern, re.IGNORECASE)
        for file_name in os.listdir(step_dir):
            match = filename_re.match(file_name)
            if not match:
                continue
            step_num = int(match.group(1))
            step_filepaths_per_text[keyword].append((step_num, file_name))
        # print(keyword + ': ' + step_filepaths_per_text[keyword])

    return step_filepaths_per_text

def compile_steps(step_dir, dir_to_save_triplets, model_name, corpus_name, threshold, window_size, text_filepath):

    all_full, all_drops, all_adds = [], [], []

    if not os.path.isdir(dir_to_save_triplets): os.mkdir(dir_to_save_triplets)

    files_by_type = map_text_to_step_outputs(model_name, corpus_name, step_dir, threshold, window_size, text_filepath)
    counter = 0

    for text_type, files in files_by_type.items():

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

            rows_all.append({
                "text_id": counter,
                "text_type": text_type,
                "output_step": step - 1,
                "current_word": data["text"].split()[-1],
                "triplet_impacted": "yes" if impacted else "no",
                "current_text": data["text"],
                "total_triplets": simplified_triplets,
                "triplet_scores": triplet_scores
            })

            new_trips = [t for t in added_trips if t not in seen_ever]

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

            prev_set = curr_set

        counter += 1

        df_full = (pd.DataFrame(rows_all).sort_values(["text_id", "output_step"]).reset_index(drop=True))
        additions_df = pd.DataFrame(rows_add).sort_values("output_step").reset_index(drop=True)
        deletions_df = pd.DataFrame(rows_drop).sort_values("output_step").reset_index(drop=True)

        df_full.to_csv(f"{dir_to_save_triplets}/full_{model_name}_{corpus_name}_{text_type}.csv", index=False)
        additions_df.to_csv(f"{dir_to_save_triplets}/additions_{model_name}_{corpus_name}_{text_type}.csv", index=False)
        deletions_df.to_csv(f"{dir_to_save_triplets}/deletions_{model_name}_{corpus_name}_{text_type}.csv", index=False)

        all_full.append(df_full)
        all_adds.append(additions_df)
        all_drops.append(deletions_df)

    all_full_df = pd.concat(all_full, ignore_index=True)
    all_adds_df = pd.concat(all_adds, ignore_index=True)
    all_drops_df = pd.concat(all_drops, ignore_index=True)

    all_full_df.to_csv(f"{dir_to_save_triplets}/full_{model_name}_{corpus_name}.csv", index=False)
    all_adds_df.to_csv(f"{dir_to_save_triplets}/additions_{model_name}_{corpus_name}.csv", index=False)
    all_drops_df.to_csv(f"{dir_to_save_triplets}/deletions_{model_name}_{corpus_name}.csv", index=False)

def add_triplets_to_eye_data(corpus_name, eye_df, triplets_df):

    triplets_df.rename(columns={"output_step": "ianum"}, inplace=True)

    if corpus_name == 'meco':

        df = pd.merge(eye_df, triplets_df[['text_id', 'ianum', 'total_triplets', 'triplet_scores']], how='left', on=['text_id', 'ianum'])

    elif corpus_name == 'onestop':
        pass

    else:
        raise NotImplementedError(f'Corpus {corpus_name} not implemented. Choose between "meco", "onestop"')

    # add number of triplets
    df['n_triplets'] = [0 if triplets == '[]' or pd.isna(triplets) else len(triplets.split('),')) for triplets in df['total_triplets'].tolist()]
    # add summed scores
    df['triplet_scores'] = df['triplet_scores'].apply(lambda x: x.replace('[', '').replace(']', '').split(', '))
    df['sum_scores'] = [sum([float(score.strip()) for score in triplets]) if any(triplets) else 0 for triplets in df['triplet_scores'].tolist()]

    return df

def main():

    corpus_name = 'meco'
    eye_filepath = f'../data/processed/{corpus_name}_eye_mov.csv'
    step_dir = f'../data/output/step_outputs_{corpus_name}/_2025_07_31_14-54-48'
    model_name = 'relik-cie-xl'
    threshold = '0.1'
    window_size = '128'
    text_filepath = f'../data/processed/{corpus_name}_texts.csv'
    dir_to_save_triplets = f'../data/output/all_outputs_{corpus_name}'
    dir_to_save_final_data = f'../data/output/eye_data_plus_triplets_{corpus_name}.csv'

    # read in eye mov data
    eye_df = pd.read_csv(eye_filepath)

    # compile all texts
    # TODO change function to include onestop variables
    # compile_steps(step_dir, dir_to_save_triplets, model_name, corpus_name, threshold, window_size, text_filepath)
    triplets_df = pd.read_csv(f"{dir_to_save_triplets}/full_{model_name}_{corpus_name}.csv")

    # check alignment between eye mov data and triplet data
    # check_alignment(corpus_name, triplets_df, eye_df)

    # TODO check Insects Could be the Planets Next Food Source (Adv) bcs of tokenization inconsistency.
    # add total_triplets and n_triplets to eye data
    final_df = add_triplets_to_eye_data(corpus_name, eye_df, triplets_df)
    final_df.to_csv(dir_to_save_final_data)

if __name__ == '__main__':
    main()