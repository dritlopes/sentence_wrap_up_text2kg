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
data <- read.csv("onestop_eye_mov_plus_triplets_gpt-4o-mini.csv") # eye_data_plus_triplets_onestop.csv
# data <- data[data$difficulty_level == 'Adv',]
# data <- data[data$n_triplets <= 5,]

# standardize predictors with z-score
data$norm_word_pos <- scale(data$norm_word_pos, center = TRUE, scale = TRUE)
data$abs_word_pos <- scale(data$abs_word_pos, center = TRUE, scale = TRUE)
data$sent_length <- scale(data$sent_length, center = TRUE, scale = TRUE)
data$length <- scale(data$word_length_no_punctuation, center = TRUE, scale = TRUE)
data$frequency <- scale(data$wordfreq_frequency, center = TRUE, scale = TRUE)
data$surprisal <- scale(data$gpt2_surprisal, center = TRUE, scale = TRUE)
data$ianum <- scale(data$ianum, center = TRUE, scale = TRUE)
data$sentnum <- scale(data$sent_id, center = TRUE, scale = TRUE)
data$n_triplets <- scale(data$n_triplets, center = TRUE, scale = TRUE)
data$norm_ianum <- scale(data$norm_ianum, center = TRUE, scale = TRUE)

data$n_triplets_added <- scale(data$n_triplets_added, center = TRUE, scale = TRUE)
data$n_triplets_new <- scale(data$n_triplets_new, center = TRUE, scale = TRUE)
data$n_triplets_activated <- scale(data$n_triplets_activated, center = TRUE, scale = TRUE)
data$n_new_triplets_activated <- scale(data$n_new_triplets_activated, center = TRUE, scale = TRUE)

data$n_triplets_minus_one <- scale(data$n_triplets_minus_one, center = TRUE, scale = TRUE)
data$n_triplets_plus_one <- scale(data$n_triplets_plus_one, center = TRUE, scale = TRUE)

# convert . into 0 in response columns
data$first_fix_dur[data$first_fix_dur == "."] <- 0
data$first_fix_dur <- as.numeric(data$first_fix_dur)
class(data$first_fix_dur)
data$gaze_dur[data$gaze_dur == "."] <- 0
data$gaze_dur <- as.numeric(data$gaze_dur)
class(data$gaze_dur)

# 1. Number of triplets

# 1.1. First Fix Duration

# baseline
firstFixBase <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|article_text_id) + (1|participant_id), data = data)
summary(firstFixBase)
# Removed abs_word_pos because of perfect co-linearity with norm_word_pos leading to error.

# main model
firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + (1|article_text_id) + (1|participant_id), data = data)
summary(firstFix)
# relik: no effect of n of triplets
# gpt: pos effect of n of triplets

# n-1 and n+1
firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + n_triplets_minus_one + n_triplets_plus_one + (1|article_text_id) + (1|participant_id), data = data)
summary(firstFix)
# gpt: pos effect of n-1 and n+1, but negative effect of n

# interaction
firstFixInt <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets + norm_ianum * n_triplets + length + frequency + surprisal + (1|article_text_id) + (1|participant_id), data = data)
summary(firstFixInt)
# relik: positive effect of number of triplets, and negative interaction with norm_ianum
# gpt: positive effect of number of triplets, and negative interaction with norm_word_pos


# compare models
anova(firstFixBase, firstFixInt)

# 1.2. Gaze Duration

# Baseline
gazeDurBase <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|article_text_id) + (1|participant_id), data = data)
summary(gazeDurBase)

# Main model
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + (1|article_text_id) + (1|participant_id), data = data)
summary(gazeDur)
# relik: No effect of n of triplets
# gpt: pos eff of n of triplets

# n-1 and n+1
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + n_triplets_minus_one + n_triplets_plus_one + (1|article_text_id) + (1|participant_id), data = data)
summary(gazeDur)
# gpt: pos effect of n-1 and n+1, but negative effect of n

# Interaction
gazeDurInt <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets + norm_ianum * n_triplets + length + frequency + surprisal + (1|article_text_id) + (1|participant_id), data = data)
summary(gazeDurInt)
# relik: Positive effect 
# gpt: positive effect of n of triplets, neg interaction with norm_word_pos

# compare models
anova(gazeDurBase, gazeDurInt)

# non-linear effects
data$participant_id_int <- as.integer(gsub(".*_(\\d+)", "\\1", data$participant_id))
data <- data %>%mutate(difficulty_level = recode(difficulty_level,"Adv" = 2L,"Ele" = 1L)) 
gazeDurBaseGam <- gam(gaze_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(ianum) + s(sentnum) + s(participant_id_int, bs='re'), data=data, method='REML')
summary(gazeDurBaseGam)
plot(gazeDurBaseGam, seWithMean = TRUE, shift = coef(gazeDurBaseGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
gazeDurGam <- gam(gaze_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(ianum) + s(sentnum) + s(n_triplets) + s(participant_id_int, bs='re'), data=data, method='REML')
summary(gazeDurGam)
plot(gazeDurGam, seWithMean = TRUE, shift = coef(gazeDurGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
gam.check(gazeDurGam)

# 1.3. Total Reading Time

# Baseline
totalDurBase <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|article_text_id) + (1|participant_id), data = data)
summary(totalDurBase)

# Main model
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + (1|article_text_id) + (1|participant_id), data = data)
summary(totalDur)
# relik:no sig eff
# gtp: pos effect

# n-1 and n+1
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + n_triplets_minus_one + n_triplets_plus_one + (1|article_text_id) + (1|participant_id), data = data)
summary(totalDur)
# gpt: same thing

# Interaction
totalDurInt <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets + norm_ianum * n_triplets + length + frequency + surprisal + (1|article_text_id) + (1|participant_id), data = data)
summary(totalDurInt)
# relik: No sig effect
# gpt: postiive effect of n triplets, negative effect with norm_word_pos and positive effect on norm_ianum

# compare models
anova(totalDurBase, totalDurInt)

# non-linear effects
data$participant_id_int <- as.integer(gsub(".*_(\\d+)", "\\1", data$participant_id))
df <- df %>%mutate(difficulty_level = recode(difficulty_level,"Adv" = 2L,"Ele" = 1L)) 
df <- df %>%
  separate(text_id, into = c("article_batch", "article_id", "difficulty_level"), sep = "-") %>%
  mutate(across(everything(), as.integer)) %>%
  mutate(text_id_int = article_batch * 1e6 + article_id * 10 + difficulty_level)
totalDurBaseGam <- gam(total_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(ianum) + s(sentnum) + s(text_id_int, bs='re') + s(participant_id_int, bs='re'), data=data, method='REML')
summary(totalDurBaseGam)
plot(totalDurBaseGam, seWithMean = TRUE, shift = coef(totalDurBaseGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
totalDurGam <- gam(total_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(ianum) + s(sentnum) + s(n_triplets) + s(text_id_int, bs='re') + s(participant_id_int, bs='re'), data=data, method='REML')
summary(totalDurGam)
plot(totalDurGam, seWithMean = TRUE, shift = coef(totalDurGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
gam.check(totalDurGam)


# 2. Number of triplets added 

# 2.1. First Fix Dur

# baseline
firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|text_id) + (1|participant_id), data = data)
summary(firstFix)

# main model
firstFixInt <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_added + norm_ianum * n_triplets_added + length + frequency + surprisal + (1|text_id) + (1|participant_id), data = data)
summary(firstFixInt)
# no sig effect

# compare models
anova(firstFix, firstFixInt)

# 2.2. Gaze Duration

# baseline
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|text_id) + (1|participant_id), data = data)
summary(gazeDur)

# main model
gazeDurInt <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_added + norm_ianum * n_triplets_added + length + frequency + surprisal + (1|text_id) + (1|participant_id), data = data)
summary(gazeDurInt)
# tiny positive effect

# compare models
anova(gazeDur, gazeDurInt)

# 2.3. Total Reading Time

# baseline
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|text_id) + (1|participant_id), data = data)
summary(totalDur)

# main model
totalDurInt <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_added + norm_ianum * n_triplets_added + length + frequency + surprisal + (1|text_id) + (1|participant_id), data = data)
summary(totalDurInt)
# tiny positive effect

# compare models
anova(totalDur, totalDurInt)

# 3. Number of new triplets added

# 3.1. First Fixation Duration

# baseline
firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|text_id) + (1|participant_id), data = data)
summary(firstFix)

# main model
firstFixInt <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_new + norm_ianum * n_triplets_new + length + frequency + surprisal + (1|text_id) + (1|participant_id), data = data)
summary(firstFixInt)

# 3.2. Gaze Duration

# baseline
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|text_id) + (1|participant_id), data = data)
summary(gazeDur)

# main model
gazeDurInt <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_new + norm_ianum * n_triplets_new + length + frequency + surprisal + (1|text_id) + (1|participant_id), data = data)
summary(gazeDurInt)

# 3.3. Total Reading Time

# baseline
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|text_id) + (1|participant_id), data = data)
summary(totalDur)

# main model
totalDurInt <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_new + norm_ianum * n_triplets_new + length + frequency + surprisal + (1|text_id) + (1|participant_id), data = data)
summary(totalDurInt)


# 4. Number of activated triplets
# No sig effects

# 4.1. First Fixation Duration

# baseline
firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|text_id) + (1|participant_id), data = data)
summary(firstFix)

# main model
firstFixInt <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_activated + norm_ianum * n_triplets_activated + length + frequency + surprisal + (1|text_id) + (1|participant_id), data = data)
summary(firstFixInt)

# 4.2. Gaze Duration

# baseline
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|text_id) + (1|participant_id), data = data)
summary(gazeDur)

# main model
gazeDurInt <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_activated + norm_ianum * n_triplets_activated + length + frequency + surprisal + (1|text_id) + (1|participant_id), data = data)
summary(gazeDurInt)

# 4.3. Total Reading Time

# baseline
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|text_id) + (1|participant_id), data = data)
summary(totalDur)

# main model
totalDurInt <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_activated + norm_ianum * n_triplets_activated + length + frequency + surprisal + (1|text_id) + (1|participant_id), data = data)
summary(totalDurInt)


# 5. sanity checks
# co-relation between n of triplets and word position in text
cor.test(data$n_triplets, data$ianum)
# co-relation between n of triplets and word position in sentence
cor.test(data$n_triplets, data$norm_word_pos)
# co-relation between n of triplets and sum scores
cor.test(data$n_triplets, data$sum_scores)
# correlation between n of triplets and surprisal
cor.test(data$n_triplets_added, data$surprisal)

# 6. isolate last sentence and add interaction between n_triplets and norm_word_pos
data_last_sentence <- data[data$norm_sentnum == 1,]
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * n_triplets + length + frequency + surprisal + ianum + (1|text_id)  + (1|participant_id), data = data_last_sentence)
summary(gazeDur)
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * n_triplets + length + frequency + surprisal + ianum + (1|text_id) + (1|participant_id), data = data_last_sentence)
summary(totalDur)
