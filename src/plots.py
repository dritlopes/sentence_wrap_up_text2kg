import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/output/eye_data_plus_triplets_meco.csv')

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
def label_group(group):
    median_val = group['n_triplets'].median()
    group = group.copy()
    group['n_triplets_bin'] = group['n_triplets'].apply(
        lambda x: 'high' if x > median_val else 'low')
    return group
df['sentnum_bin'] = pd.cut(df['sentnum'], bins=10)
df = df.groupby('sentnum_bin', group_keys=False).apply(label_group)
df['norm_word_pos_bin'] = pd.cut(df['norm_word_pos'], bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
df = df.dropna(subset=['norm_word_pos_bin'])
graph = sns.relplot(x=df['norm_word_pos_bin'].astype(str), y=df['total_dur'], hue=df['n_triplets_bin'], col=df['sentnum_bin'], kind='line', col_wrap=2)
graph.tick_params(labelrotation=45)
plt.tight_layout()
plt.show()
plt.clf()

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