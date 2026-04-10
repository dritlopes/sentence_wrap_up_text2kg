# Triplet generation and eye movements in reading

This repo contains the code for the analysis reported in the paper: 

Lopes Rego, AT, Snell, J., Meeter, M. _Capturing comprehension in the reading brain: Effects from language model-derived propositions on eye movements._

## Abstract
"It is well known that reading slows down when text comprehension is difficult. Yet incorporating higher-order linguistic information remains a challenge for models of reading, as this requires representing meaning as it is dynamically derived from text. Here we propose to approximate relations that readers may infer during reading by using propositional triplets – structured representations of the form subject, relation, and object, such as “Google, test, car” in “Google has begun testing the electric car”. Drawing from natural language processing techniques for information retrieval, we extracted triplets from English texts in three corpora of eye movements and measured the effect of triplet activation on word reading times. Across corpora, increased triplet activation at a word was associated with longer reading times at the previous word, beyond the effects of position in sentence, position in text, length, frequency, and surprisal. By offering a way to model text comprehension through linguistic relationships, our approach may encourage further research into integrating semantic-level processing into models of reading behaviour.  "

---

## Resources

* Model source: <https://github.com/SapienzaNLP/relik> and <https://developers.openai.com/api/docs/models/gpt-4o-mini>
* Eye-movement corpus sources: [Provo](https://osf.io), [MECO](https://osf.io/srdhm) and [OneStop](https://osf.io/2prdq/)
* Word frequency source: [SUBTLEX-UK]()
* VU **BAZIS** HPC: <https://vu.nl/en/research/portal/research-impact-support-portal/high-performance-research-computing>

---

## Repository layout

The data folder is not available here because the files are too big. Please contact a.t.lopesrego@vu.nl if you'd like to have any of our processed datasets.

| Path / file                            | What it holds                                                                                                                                |
|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **src/process_corpus.py**              | Code that process the eye movement data files from MECO and OneStop corpora. Python file that produces the step-level JSON files             |
| **src/generate_triplets_<model>.py**   | Code that runs the text-to-triplets model on the texts from the eye movement corpora                                                         |
| **src/<model>_test_batch.sh**          | Bash script that queues `generate_triplets_<model>.py` on the BAZIS cluster                                                                  |
| **src/compile_output_<model>.py**      | Code that converts the outputs from the relik or gpt model into a csv, and merges the output with the eye movement data for further analysis |
| **src/visualisations.ipynb**           | Code for generating plots to visualise the data                                                                                              |
| **src/stats_analysis_<corpus_name>.R** | Code for running the analysis reported in the paper                                                                                          |
| **data/output**                        | Final CSVs with eye movement data and triplets generation info per word (e.g. triplets generated, number of triplets)                        |
| **data/processed**                     | Processed eye movement data from corpora MECO and OneStop                                                                                    |
| **data/raw**                           | Raw datasets from corpora and other resources (e.g. frequency resource SUBTLEX)                                                              |
| **data/analysis**                      | Output data from statistical analysis scripts in R                                                                                           |
---

## GPT model (OpenAI API)

### How to run

1. Run `process_corpus.py` to generate processed corpus data that will be have the input for the triplet generation step.

   1. Provo input file: `Provo_Corpus-Eyetracking_Data.csv`
   2. MECO input file: `joint_data_trimmed.csv`
   3. OneStop input file: `ia_Paragraph_ordinary.csv`
   4. You will need the SUBTLEX-UK frequency file if the corpus is Provo. 
   5. The output files are `<corpus>_texts.csv`, `<corpus>_words.csv`, and `<corpus>_eye_mov.csv` (see Data Files and Variables for more details)
   

2. Run `generate_triplets_gpt_api.py` to generate triplets from the corpus texts. 
   1. You will need an OpenAPI key to able to get responses from the OpenAI API.
   2. The input file is `<corpus>_texts.csv` (output file of `process_corpus.py`)
   3. The output file is `<model>_triplets_<corpus>.json` (see Data Files and Variables for more details)

3. Run `compile_output_gpt.py` to merge triplets with eye movement data at the word level. 
   1. The input files are the word file `<corpus>_words.csv`, the eye movement file `<corpus>_eye_mov.csv` (all output from process_corpus.py), and the triplet file `<model>_triplets_<corpus>_<run>.json` (output from generate_triplets_gpt_api.py)
   2. The output file is `<corpus>_eye_mov_plus_triplets_<model>.csv` with triplets merged to eye movement data. This is further processed in the R scripts for analysis.

4. Run `stats_analysis_<corpus>` in R to process statistical analyses on the data with eye movements and triplets.
    1. The input file is `<corpus>_eye_mov_plus_triplets_<model>.csv` (from `compile_output_gpt.py`).
    2. For each stat model, the output files are the model in .rds (`lmer_<reading_measure>_triplet_added_<run>.rds`), the results of the model in .csv (`lmer_<reading_measure>_triplet_added_<run>.csv`), and anova results of comparing the full model with its baseline in .csv (`anova_<reading_measure>_triplet_added_<run>.csv`).

### Data Files and Variables 

1. `<corpus>_texts.csv` (from process_corpus.py)
   
| Variable (column name)                  | Definition                                                  |
|-----------------------------------------|-------------------------------------------------------------|
| `text_id` if corpus is MECO or Provo    | ID of text                                                  |
| `article_batch` if corpus is OneStop    | Batch of article paragraph belongs to                       |
| `article_id` if corpus is OneStop       | ID of article paragraph belongs to                          |
| `paragraph_id` if corpus is OneStop     | ID of paragraph                                             |
| `difficulty_level` if corpus is OneStop | Readability version of the article the paragraph belongs to |
| `keyword` if corpus is MECO or Provo    | Keyword for text                                            |
| `text` if corpus is MECO or Provo       | Text                                                        |
| `paragraph` if corpus is OneStop        | Paragraph                                                   |

2. `<corpus>_words.csv` (from process_corpus.py)

| Variable (column name)                  | Definition                                                  |
|-----------------------------------------|-------------------------------------------------------------|
| `text_id`                               | ID of text                                                  |
| `article_batch` if corpus is OneStop    | Batch of article paragraph belongs to                       |
| `article_id` if corpus is OneStop       | ID of article paragraph belongs to                          |
| `article_title` if corpus is OneStop    | Title of article paragraph belongs to                       |
| `paragraph_id` if corpus is OneStop     | ID of paragraph                                             |
| `difficulty_level` if corpus is OneStop | Readability version of the article the paragraph belongs to |
| `keyword` if corpus is MECO and Provo   | Keyword for text                                            |
| `text` if corpus is MECO or Provo       | Text                                                        |
| `paragraph` if corpus is OneStop        | Paragraph                                                   |
| `ianum`                                 | ID of word (= position in text)                             |
| `ia`                                    | Word                                                        |
| `article_ianum` if corpus is OneStop    | ID of word in article the paragraph belongs to              |
| `sent_id`                               | ID of sentence the word belongs to                          |
| `sent_length`                           | Length of sentence in words                                 |
| `abs_word_pos`                          | Absolute word position of word in sentence                  |
| `norm_word_pos`                         | Normalized word position of word in sentence                |
| `norm_ianum`                            | Absolute word position of word in text                      |
| `frequency`                             | Frequency of word in log zipf                               |
| `length`                                | Length of word in characters                                |
| `surprisal`                             | Surprisal of word in bits                                   |
| `pos_tag`                               | Universal part-of-speech tag of word                        |


3. `<corpus>_eye_mov.csv` (from process_corpus.py): all variables from `<corpus>_words.csv` plus the following variables from eye movements data:

| Variable (column name) | Definition                                           |
|------------------------|------------------------------------------------------|
| `participant_id`       | ID of participant that generated the eye-movements   |
| `first_fix_dur`        | Duration of first fixation during first-pass at word |
| `gaze_dur`             | Total duration of all first-pass fixations at word   |
| `total_dur`            | Total duration of all fixations at word              |

4. `<model>_triplets_<corpus>_<run>.json` (from generate_triplets_gpt_api.py)

| Variable (json key)  | Definition                                                                         |
|----------------------|------------------------------------------------------------------------------------|
| `text_id`            | ID of text                                                                         |
| `extracted_triplets` | List of dictionaries with extracted triplets from the text                         |
| `step`               | ID of model triplet generation iteration                                           |
| `context`            | The input string to the model at current step                                      |
| `triplets`           | List of dictionaries with extracted triplets from the text at current step         |
| `entity_1`           | Dictionary with label and mention of first entity in the triplet                   |
| `relation`           | Dictionary with label and mention of relation in the triplet                       |
| `entity_2`           | Dictionary with label and mention of second entity in the triplet                  |
| `label`              | The (unique) name of the entity/relation                                           |
| `mention`            | The text span in the input text of current step that refers to the entity/relation |

5. `<corpus>_eye_mov_plus_triplets_<model>_<run>.csv` (from compile_output_gpt.py)

| Variable (column name) | Definition                                         |
|------------------------|----------------------------------------------------|
| `participant_id`       | ID of participant that generated the eye-movements |
| `text_id`              | ID of text                                         |
| `ia`                   | Word                                               |
| `ianum`                | ID of word                                         |
| `norm_word_pos`        | Normalized word position of word in sentence       |
| `norm_ianum`           | Absolute word position of word in text             |
| `frequency`            | Frequency of word in log zipf                      |
| `length`               | Length of word in characters                       |
| `surprisal`            | Surprisal of word in bits                          |
| `triplet_added`        | Whether a triplet has been formed                  |


## Relik model (Huggingface)

### Output file-naming pattern

1. When running the script `generate_triplets_relik.py`, a json file is generated per word in each text, with the triplets generated by the model up to each word. 
This step output filename follows the format `output_step_<word_position_in_text>_<model>_<corpus>_<text_identifier>_<model_threshold>_<model_window_size>.json`.
* `word_position_in_text` is the output step number which is equivalent to until where in the text (word-wise) the input of the model goes. 
* `model` is the name of the relik model implementation in HuggingFace (e.g. `relik-cie-small`, `relik-cie-large`, `relik-cie-xl`).
* `corpus` is the name of the eye movement corpus (`meco` or `onestop`).
* `text_identifier` is a unique code that identifies each text of the corpus. For OneStop, the identifier has the format 
`<article_bath-article_id-paragraph_id-difficulty_level>`. For MECO, the identifier has the format `<most_frequent_proper_noun_in_text>`.
* `model_threshold` is the score threshold for a triplet to be considered sufficiently recognized in the input.
* `model_window_size` is the number of characters in the input to be processed as one chunk by the text-to-triplet model (Relik).

2. When running the script `compile_output_relik.py`, three csv files are generated per text. The filename follows the format 
`<type_of_output>_<model>_<corpus>_<text_identifier>.csv`
* `type of output` is the type of output, which can `additions` (only the steps (word positions) in which new triplets were added to the output), 
`deletions` (only steps (word positions) in which triplets were removed from the output) or `full` (all output steps).
Example: `additions_relik-cie-large_meco_beekeeping.csv`

### How the output CSV looks like

The full CSV produced by `compile_output_relik.py`, before merging with eye movement data:

| Column                               | Meaning                                                                                                  |
|--------------------------------------|----------------------------------------------------------------------------------------------------------|
| `text_id` if corpus MECO or Provo    | ID of the corpus text                                                                                    |
| `article_batch` if corpus OneStop    | Batch of article paragraph belongs to                                                                    |
| `article_id` if corpus OneStop       | ID of article paragraph belongs to                                                                       |
| `paragraph_id` if corpus OneStop     | ID of paragraph                                                                                          |
| `difficulty_level` if corpus OneStop | readability version of the article the paragraph belongs to                                              |
| `output_step`                        | Word index                                                                                               |
| `current_word`                       | The word that triggered this output                                                                      |
| `current_text`                       | Full text up to `current_word`                                                                           |
| `triplets_impacted`                  | The number of triplets that were either added or dropped compared to previous step                       |
| `total_triplets`                     | Complete triplet set generated at this step                                                              |
| `triplet_scores`                     | Relik scores for generated triplets                                                                      |
| `n_triplets_added`                   | The number of triplets added at this step                                                                |
| `n_triplets_removed`                 | The number of triplets removed at this step                                                              |
| `n_triplets`                         | The total number of triplets at this step                                                                |
| `n_triplets_new`                     | The number of new triplets added at this step (that have never appeared in the output of previous steps) |

---

## References

Berzak, Y., Malmaud, J., Shubi, O., Meiri, Y., Lion, E., & Levy, R. (2025). Onestop: A 360-participant english eye-tracking dataset with different reading regimes. PsyArXiv preprint.
Siegelman, N., Schroeder, S., Acartürk, C., Ahn, H. D., Alexeeva, S., Amenta, S., ... & Kuperman, V. (2022). Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement Corpus (MECO). Behavior research methods, 54(6), 2843-2863.

## Acknowledgements

Big thanks to the Research Assistants [Konstantin Mihhailov](https://github.com/ElectricBoogaloo6) and [Haomin Wu](https://github.com/returnhw99), for conducting hyperparameter searching with Relik, and helping to set up the analysis, respectively.