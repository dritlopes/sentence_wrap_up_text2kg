library(lme4)
library(readr)
library(ggplot2)
library(dplyr)
library(lmerTest)
library(mgcv)
library(glue)

# clear existing workspace objects 
rm(list = ls())
# set working directory to where the data file is located & results should be saved
setwd(glue("/Users/adriellilopes/PycharmProjects/Text2KG/data/output"))

# read in data
data <- read.csv("eye_data_plus_triplets_meco.csv")

# standardize predictors with z-score
data$norm_word_pos <- scale(data$norm_word_pos, center = TRUE, scale = TRUE)
data$abs_word_pos <- scale(data$abs_word_pos, center = TRUE, scale = TRUE)
data$sent_length <- scale(data$sent_length, center = TRUE, scale = TRUE)
data$length <- scale(data$length, center = TRUE, scale = TRUE)
data$frequency <- scale(data$frequency, center = TRUE, scale = TRUE)
data$surprisal <- scale(data$surprisal, center = TRUE, scale = TRUE)
data$ianum <- scale(data$ianum, center = TRUE, scale = TRUE)
data$sentnum <- scale(data$sentnum, center = TRUE, scale = TRUE)
data$n_triplets <- scale(data$n_triplets, center = TRUE, scale = TRUE)
data$norm_ianum <- scale(data$norm_ianum, center = TRUE, scale = TRUE)
data$sum_scores <- scale(data$sum_scores, center = TRUE, scale = TRUE)

# first fix dur - baseline model
firstFixBase <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + ianum + (1|participant_id), data = data)
summary(firstFixBase)
# AL: removed abs_word_pos because of perfect co-linearity with norm_word_pos leading to error.
# first fix dur - main model
firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + ianum + n_triplets + (1|participant_id), data = data)
summary(firstFix)
firstFixInt <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets + length + frequency + surprisal + ianum + (1|participant_id), data = data)
summary(firstFixInt)

# gaze dur - baseline model
gazeDurBase <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + ianum + (1|participant_id), data = data)
summary(gazeDurBase)
# gaze dur - main model
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + ianum + n_triplets + (1|participant_id), data = data)
summary(gazeDur)
# AL: n of triplets positive effect; word position in sentence negative effect; word position in text negative effect 
# AL: sum scores very similar effect, slightly weaker
# compare models
anova(gazeDurBase, gazeDur)
# gaze dur - interaction n_triplets
# AL: no interaction between word position in sentence and n of triplets; same for sum_scores
gazeDurInt <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets + length + frequency + surprisal + ianum + (1|participant_id), data = data)
summary(gazeDurInt)
# non-linear effects
# AL: normalized ianum no wrap up
data$participant_id_int <- as.integer(gsub(".*_(\\d+)", "\\1", data$participant_id))
gazeDurBaseGam <- gam(gaze_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(ianum) + s(sentnum) + s(participant_id_int, bs='re'), data=data, method='REML')
summary(gazeDurBaseGam)
plot(gazeDurBaseGam, seWithMean = TRUE, shift = coef(gazeDurBaseGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
gazeDurGam <- gam(gaze_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(ianum) + s(sentnum) + s(n_triplets) + s(participant_id_int, bs='re'), data=data, method='REML')
summary(gazeDurGam)
plot(gazeDurGam, seWithMean = TRUE, shift = coef(gazeDurGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
gam.check(gazeDurGam)

# total reading time - baseline model
totalDurBase <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + ianum + (1|participant_id), data = data)
summary(totalDurBase)
# total reading time - main model
# AL: same effects as gaze dur
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + ianum + n_triplets + (1|participant_id), data = data)
summary(totalDur)
# compare models
anova(totalDurBase, totalDur)
# total dur - interaction n_triplets
# AL: no interaction between word position in sentence and n of triplets
totalDurInt <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets + length + frequency + surprisal + ianum + (1|participant_id), data = data)
summary(totalDurInt)
# non-linear effects
# AL: normalized ianum no wrap up
data$participant_id_int <- as.integer(gsub(".*_(\\d+)", "\\1", data$participant_id))
totalDurBaseGam <- gam(total_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(ianum) + s(sentnum) + s(participant_id_int, bs='re'), data=data, method='REML')
summary(totalDurBaseGam)
plot(totalDurBaseGam, seWithMean = TRUE, shift = coef(totalDurBaseGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
totalDurGam <- gam(total_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(ianum) + s(sentnum) + s(n_triplets) + s(participant_id_int, bs='re'), data=data, method='REML')
summary(totalDurGam)
plot(totalDurGam, seWithMean = TRUE, shift = coef(totalDurGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
gam.check(totalDurGam)

# sanity checks
# AL: strong positive correlation with word pos in text, but weak positive correlation with word pos in sentence.
# co-relation between n of triplets and word position in text
cor.test(data$n_triplets, data$ianum)
# co-relation between n of triplets and word position in sentence
cor.test(data$n_triplets, data$norm_word_pos)
# co-relation between n of triplets and sum scores
cor.test(data$n_triplets, data$sum_scores)

# isolate last sentence and add interaction between n_triplets and norm_word_pos
# AL: only in gaze dur, sig neg interaction between n of triplets and word position in sentence (less acceleration along sentence with more triplets)
data_last_sentence <- data[data$norm_sentnum == 1,]
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * n_triplets + length + frequency + surprisal + ianum + (1|participant_id), data = data_last_sentence)
summary(gazeDur)
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * n_triplets + length + frequency + surprisal + ianum + (1|participant_id), data = data_last_sentence)
summary(totalDur)
