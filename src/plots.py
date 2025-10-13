import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# LENGTH OF TEXT
# df = df[df['norm_ianum']==1.0]
# print(np.mean(df['ianum'].tolist()), np.std(df['ianum'].tolist()))
# graph = sns.displot(data=df, x='ianum', stat='probability')
# graph.set_axis_labels("Text length in OneStop")
# plt.show()
# plt.clf()

# NUMBER OF TRIPLETS
# number of triplets per word position in sentence
# df = pd.read_csv('../data/output/eye_data_plus_triplets_onestop_article.csv')
# df.dropna(subset=['n_triplets'], inplace=True)
# df['norm_word_pos_bin'] = pd.cut(df['norm_word_pos'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# graph = sns.catplot(data=df, x='norm_word_pos_bin', y='n_triplets', kind='bar')
# graph.set_axis_labels("Normalized word position in sentence", "Number of triplets")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()
# df = pd.read_csv('../data/output/eye_data_plus_triplets_onestop.csv')
# df['norm_word_pos_bin'] = pd.cut(df['norm_word_pos'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# graph = sns.catplot(data=df, x='norm_word_pos_bin', y='n_triplets', kind='bar')
# graph.set_axis_labels("Normalized word position in sentence", "Number of triplets")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()
# number of triplets per word position in paragraph
# df = pd.read_csv('../data/output/eye_data_plus_triplets_onestop_article.csv')
# df.dropna(subset=['n_triplets'], inplace=True)
# df['norm_ianum_bin'] = pd.cut(df['norm_ianum'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# graph = sns.catplot(data=df, x='norm_ianum_bin', y='n_triplets', kind='bar')
# graph.set_axis_labels("Normalized word position in paragraph", "Number of triplets")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()
# df = pd.read_csv('../data/output/eye_data_plus_triplets_onestop.csv')
# df['norm_ianum_bin'] = pd.cut(df['norm_ianum'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# graph = sns.catplot(data=df, x='norm_ianum_bin', y='n_triplets', kind='bar')
# graph.set_axis_labels("Normalized word position in paragraph", "Number of triplets")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()

# # number of triplets distribution per article id per difficulty level
# df = pd.read_csv('../data/output/all_outputs_onestop/paragraph_level/full_relik-cie-xl_onestop.csv')
# df['article_id'] = [f'{batch}-{id}' for batch, id in zip(df['article_batch'].tolist(), df['article_id'].tolist())]
# graph = sns.catplot(data=df, x='n_triplets', y='article_id', hue='difficulty_level')
# graph.set_axis_labels("Number of triplets in OneStop", "Article ID")
# # plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()
# number of triplets per word position in sentence per difficulty level
# df = pd.read_csv('../data/output/eye_data_plus_triplets_onestop.csv')
# df['norm_word_pos'] = pd.cut(df['norm_word_pos'], bins=10)
# graph = sns.catplot(data=df, x='norm_word_pos', y='n_triplets', hue='difficulty_level', kind='bar', height=8, aspect=1.2)
# graph.set_axis_labels("Normalized Word Position in Sentence", "Number of Triplets")
# plt.xticks(rotation=45)
# graph.tight_layout(pad=1)
# plt.show()
# plt.clf()
# number of triplets per word position in paragraph per difficulty level
# df = pd.read_csv('../data/output/eye_data_plus_triplets_onestop.csv')
# df['norm_ianum'] = pd.cut(df['norm_ianum'], bins=10)
# graph = sns.catplot(data=df, x='norm_ianum', y='n_triplets', hue='difficulty_level', kind='bar', height=8, aspect=1.2)
# graph.set_axis_labels("Normalized Word Position in Paragraph", "Number of Triplets")
# plt.xticks(rotation=45)
# graph.tight_layout(pad=1)
# plt.show()
# plt.clf()

# # distribution of number of triplets per word position in article
# df = pd.read_csv('../data/output/all_outputs_onestop/article_level/full_relik-cie-xl_onestop.csv')
# df['ianum_bin'] = pd.cut(df['output_step'], bins=10)
# df['n_triplets'] = [0 if triplets == '[]' or pd.isna(triplets) else len(triplets.split('),')) for triplets in df['total_triplets'].tolist()]
# graph = sns.catplot(data=df, x='ianum_bin', y='n_triplets', kind='bar')
# graph.set_axis_labels("Word position in article", "Number of triplets")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()

# distribution of number of triplets per word position in paragraph
# # df = pd.read_csv('../data/output/all_outputs_onestop/paragraph_level/full_relik-cie-xl_onestop.csv')
# df = pd.read_csv('../data/output/all_outputs_meco/full_relik-cie-xl_meco.csv')
# # df = df[(df['article_batch'] == 1) & (df['article_id'] == 0) & (df['difficulty_level'] == 'Adv') & (df['paragraph_id']==2)]
# df['ianum_bin'] = pd.cut(df['output_step'], bins=10)
# graph = sns.catplot(data=df, x='ianum_bin', y='n_triplets', kind='bar')
# graph.set_axis_labels("Word position in paragraph in MECO", "Number of triplets")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()

# distribution of number of triplets per word position
# df = pd.read_csv('../data/output/eye_data_plus_triplets_meco.csv')
# df['norm_ianum_bin'] = pd.cut(df['norm_ianum'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# graph = sns.catplot(data=df, x='norm_ianum_bin', y='n_triplets', kind='bar')
# graph.set_axis_labels("Normalized word position in text", "Number of triplets")
# graph.fig.suptitle('MECO')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()
# df['norm_word_pos'] = pd.cut(df['norm_word_pos'], bins=10)
# graph = sns.catplot(data=df, x='norm_word_pos', y='n_triplets', kind='bar')
# graph.set_axis_labels("Word position in sentence", "Number of triplets")
# graph.fig.suptitle('MECO')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()

# WORD POSITION IN SENTENCE

# # plot norm word position in sentence and total duration
# AL: very uniform, except for longer reading time in the very beginning of text
# df['norm_word_pos_bin'] = pd.cut(df['norm_word_pos'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# graph = sns.catplot(data=df, x='norm_word_pos_bin', y='total_dur', kind='bar')
# graph.set_axis_labels("Normalized word position in sentence", "Total reading time")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# graph.savefig('../data/output/plot_pos_sent_total_dur.tiff', dpi=300, format='tiff')
# plt.clf()

# # plot norm word position in sentence and total duration by sentence number
# AL: most sentences show small increase in reading times torwards the end of the sentence
# df['norm_word_pos_bin'] = pd.cut(df['norm_word_pos'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# df = df.dropna(subset=['norm_word_pos_bin'])
# df['sentnum_bin'] = pd.cut(df['sentnum'], bins=10)
# graph = sns.relplot(x=df['norm_word_pos_bin'].astype(str), y=df['total_dur'], col=df['sentnum_bin'], kind='line', col_wrap=2)
# graph.tick_params(labelrotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()

# # plot norm word position in sentence and total duration by sentence length
# AL: big wrap up at very short sentences, and smaller wrap up in the rest
# df['norm_word_pos_bin'] = pd.cut(df['norm_word_pos'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# df = df.dropna(subset=['norm_word_pos_bin'])
# df['sent_length_bin'] = pd.cut(df['sent_length'], bins=10)
# graph = sns.relplot(x=df['norm_word_pos_bin'].astype(str), y=df['total_dur'], col=df['sent_length_bin'], kind='line', col_wrap=2)
# graph.tick_params(labelrotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()

# # plot norm word position in sentence and total duration by n of triplets (hue) and by sentence number (col)
# AL:
# def label_group(group):
#     median_val = group['n_triplets'].median()
#     group = group.copy()
#     group['n_triplets_bin'] = group['n_triplets'].apply(
#         lambda x: 'high' if x > median_val else 'low')
#     return group
# df['sentnum_bin'] = pd.cut(df['sentnum'], bins=10)
# df = df.groupby('sentnum_bin', group_keys=False).apply(label_group)
# df['norm_word_pos_bin'] = pd.cut(df['norm_word_pos'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# df = df.dropna(subset=['norm_word_pos_bin'])
# graph = sns.relplot(x=df['norm_word_pos_bin'].astype(str), y=df['total_dur'], hue=df['n_triplets_bin'], col=df['sentnum_bin'], kind='line', col_wrap=2)
# graph.tick_params(labelrotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()

# plot only word positions and total durations belonging to last sentence by n of triplets
# AL: wrap up in all n of triplets, but longer durations when more triplets
# df = df[df['norm_sentnum'] == 1]
# df['norm_word_pos_bin'] = pd.cut(df['norm_word_pos'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# df = df.dropna(subset=['norm_word_pos_bin'])
# df['n_triplets_bin'] = pd.cut(df['n_triplets'], bins=3)
# graph = sns.relplot(x=df['norm_word_pos_bin'].astype(str), y=df['total_dur'], col=df['n_triplets_bin'], kind='line')
# graph.tick_params(labelrotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()

# WORD POSITION IN TEXT

# # plot absolute word position in text and total duration
# # AL: slight increase in reading times in the last bin
# df['ianum_bin'] = pd.cut(df['ianum'], bins=10)
# graph = sns.catplot(data=df, x='ianum_bin', y='total_dur', kind='bar')
# graph.set_axis_labels("Word position in text", "Total reading time")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()

# # plot norm word position in text and total duration
# # AL: wrap up disappears when normalizing word position in text (maybe only for longer texts?)
# df['norm_ianum_bin'] = pd.cut(df['norm_ianum'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
# graph = sns.catplot(data=df, x='norm_ianum_bin', y='total_dur', kind='bar')
# graph.set_axis_labels("Normalized word position in text", "Total reading time")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()

# # plot norm word position in text and total duration by text length
# # AL: wrap up only for longest texts and.  medium-short texts (?)
# df['norm_ianum_bin'] = pd.cut(df['norm_ianum'], bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
# df = df.dropna(subset=['norm_ianum_bin'])
# df = df.merge(df.groupby(['participant_id','text_id'])['ianum'].max().rename('text_length'), on='text_id')
# df['text_length_bin'] = pd.cut(df['text_length'], bins=5)
# graph = sns.relplot(x=df['norm_ianum_bin'].astype(str), y=df['total_dur'], col=df['text_length_bin'], kind='line', col_wrap=2)
# graph.tick_params(labelrotation=45)
# plt.tight_layout()
# plt.show()
# plt.clf()