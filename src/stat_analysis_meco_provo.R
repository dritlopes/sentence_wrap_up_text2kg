# install.packages("broom.mixed")

library(lme4)
library(readr)
library(ggplot2)
library(dplyr)
library(lmerTest)
library(mgcv)
library(glue)
library(broom.mixed)

# clear existing workspace objects 
rm(list = ls())
# set working directory to where the data file is located & results should be saved
setwd("/Users/adriellilopes/PycharmProjects/Text2KG/data/")

# read in data
corpus <- 'meco'
model <- 'gpt-4o-mini'

for (run in 1:10) {
  
  data <- read.csv(glue("output/{model}/{corpus}/{corpus}_eye_mov_plus_triplets_{model}.csv")) 
  # data <- data[data$n_triplets <= 5,]
  data <- data %>% filter(`run_id` == run)
  
  # add triplets n-1 and n+1 variables
  data <- data %>% group_by(participant_id, text_id) %>% mutate(triplets_added_minus_one = if_else(lag(`n_triplets`, 1) > lag(`n_triplets`, 2), 1, 0)) %>%  ungroup()
  data <- data %>%  group_by(participant_id, text_id) %>% mutate(triplets_added = if_else(`n_triplets` > lag(`n_triplets`, 1), 1, 0)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(triplets_added_plus_one = if_else(lead(`n_triplets`, 1) > `n_triplets`, 1, 0)) %>%  ungroup()

  # standardize predictors with z-score
  # so beta coefficient will mean the amount of increase or decrease of dependent variable associated with 1 standard deviation increase of independent variable
  data$norm_word_pos <- scale(data$norm_word_pos, center = TRUE, scale = TRUE)
  data$abs_word_pos <- scale(data$abs_word_pos, center = TRUE, scale = TRUE)
  data$sent_length <- scale(data$sent_length, center = TRUE, scale = TRUE)
  data$length <- scale(data$length, center = TRUE, scale = TRUE)
  data$frequency <- scale(data$frequency, center = TRUE, scale = TRUE)
  data$surprisal <- scale(data$surprisal, center = TRUE, scale = TRUE)
  data$ianum <- scale(data$ianum, center = TRUE, scale = TRUE)
  data$norm_ianum <- scale(data$norm_ianum, center = TRUE, scale = TRUE)
  data$sentnum <- scale(data$sentnum, center = TRUE, scale = TRUE)
  data$n_triplets <- scale(data$n_triplets, center = TRUE, scale = TRUE)
  data$triplets_added <- scale(data$triplets_added, center = TRUE, scale = TRUE)
  data$triplets_added_minus_one <- scale(data$triplets_added_minus_one, center = TRUE, scale = TRUE)
  data$triplets_added_plus_one <- scale(data$triplets_added_plus_one, center = TRUE, scale = TRUE)

  # data$n_triplets_added <- scale(data$n_triplets_added, center = TRUE, scale = TRUE)
  # data$n_triplets_new <- scale(data$n_triplets_new, center = TRUE, scale = TRUE)
  # data$n_triplets_activated <- scale(data$n_triplets_activated, center = TRUE, scale = TRUE)
  # data$n_new_triplets_activated <- scale(data$n_new_triplets_activated, center = TRUE, scale = TRUE)
  
  # # 1. Number of triplets
  # 
  # # 1.1. First Fix Duration
  # 
  # # baseline model
  # firstFixBase <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
  # # summary(firstFixBase)
  # # AL: removed abs_word_pos because of perfect co-linearity with norm_word_pos leading to error.
  # 
  # # main model
  # firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + (1|participant_id), data = data)
  # summary(firstFix)
  # 
  # # save out results
  # tidy_model <- tidy(firstFix)
  # write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_{run}.csv"), row.names = FALSE)
  # 
  # # compare models
  # anova_result <- anova(firstFixBase, firstFix)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_firstFix_{run}.csv"), row.names = TRUE)
  # 
  # # main model interaction
  # firstFixInt <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets + norm_ianum * n_triplets + length + frequency + surprisal + (1|participant_id), data = data)
  # # summary(firstFixInt)
  # 
  # # save out results
  # tidy_model <- tidy(firstFixInt)
  # write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFixInt_{run}.csv"), row.names = FALSE)
  # 
  # # compare models
  # anova_result <- anova(firstFixBase, firstFixInt)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_firstFixInt_{run}.csv"), row.names = TRUE)
  # 
  # # 1.2. Gaze Duration
  # 
  # # baseline model
  # gazeDurBase <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
  # # summary(gazeDurBase)
  # 
  # # main model
  # gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + (1|participant_id), data = data)
  # # summary(gazeDur)
  # 
  # # save out results
  # tidy_model <- tidy(gazeDur)
  # write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_{run}.csv"), row.names = FALSE)
  # 
  # # compare models
  # anova_result <- anova(gazeDurBase, gazeDur)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_gazeDur_{run}.csv"), row.names = TRUE)
  # 
  # # interaction n_triplets
  # gazeDurInt <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets + norm_ianum * n_triplets + length + frequency + surprisal + (1|participant_id), data = data)
  # # summary(gazeDurInt)
  # 
  # # save out results
  # tidy_model <- tidy(gazeDurInt)
  # write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDurInt_{run}.csv"), row.names = FALSE)
  # 
  # # compare models
  # anova_result <- anova(gazeDurBase, gazeDurInt)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_gazeDurInt_{run}.csv"), row.names = TRUE)
  # 
  # # 1.3. Total Reading Time
  # 
  # # baseline model
  # totalDurBase <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
  # # summary(totalDurBase)
  # 
  # # main model
  # totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + (1|participant_id), data = data)
  # # summary(totalDur)
  # 
  # # save out results
  # tidy_model <- tidy(totalDur)
  # write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_{run}.csv"), row.names = FALSE)
  # 
  # # compare models
  # anova_result <- anova(totalDurBase, totalDur)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_totalDur_{run}.csv"), row.names = TRUE)
  # 
  # # interaction n_triplets
  # totalDurInt <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets + norm_ianum * n_triplets + length + frequency + surprisal + (1|participant_id), data = data)
  # summary(totalDurInt)
  # 
  # # save out results
  # tidy_model <- tidy(totalDurInt)
  # write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDurInt_{run}.csv"), row.names = FALSE)
  # 
  # # compare models
  # anova_result <- anova(totalDurBase, totalDurInt)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_totalDurInt_{run}.csv"), row.names = TRUE)
  
  # First Fix Dur n-1 and n+1
  firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + triplets_added_minus_one + triplets_added_plus_one + (1|participant_id), data = data)
  tidy_model <- tidy(firstFix)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_plusMinusOne_{run}.csv"), row.names = FALSE)
  # Gaze Dur n-1 and n+1
  gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + triplets_added_minus_one + triplets_added_plus_one + (1|participant_id), data = data)
  tidy_model <- tidy(gazeDur)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_plusMinusOne_{run}.csv"), row.names = FALSE)
  # Total Reading Time n-1 and n+1
  totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + n_triplets + triplets_added_minus_one + triplets_added_plus_one + (1|participant_id), data = data)
  tidy_model <- tidy(totalDur)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_plusMinusOne_{run}.csv"), row.names = FALSE)
}  
  
# # non-linear effects gaze duration
# # AL: normalized ianum no wrap up
# # data$participant_id <- as.integer(gsub(".*_(\\d+)", "\\1", data$participant_id))
# data$participant_id <- as.integer(sub("Sub", "", data$participant_id))
# # length(unique(data[["n_triplets"]]))
# gazeDurBaseGam <- gam(gaze_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(norm_ianum) + s(sentnum, k=4) + s(participant_id, bs='re'), data=data, method='REML')
# summary(gazeDurBaseGam)
# plot(gazeDurBaseGam, seWithMean = TRUE, shift = coef(gazeDurBaseGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
# gazeDurGam <- gam(gaze_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(norm_ianum) + s(sentnum, k=4) + s(n_triplets) + s(participant_id, bs='re'), data=data, method='REML')
# summary(gazeDurGam)
# plot(gazeDurGam, seWithMean = TRUE, shift = coef(gazeDurGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
# gam.check(gazeDurGam)

# # non-linear effects total reading time
# # AL: normalized ianum no wrap up
# # data$participant_id_int <- as.integer(gsub(".*_(\\d+)", "\\1", data$participant_id))
# totalDurBaseGam <- gam(total_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(ianum) + s(sentnum) + s(participant_id_int, bs='re'), data=data, method='REML')
# summary(totalDurBaseGam)
# plot(totalDurBaseGam, seWithMean = TRUE, shift = coef(totalDurBaseGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
# totalDurGam <- gam(total_dur ~ s(norm_word_pos) + s(sent_length) + s(length) + s(frequency) + s(surprisal) + s(ianum) + s(sentnum) + s(n_triplets) + s(participant_id_int, bs='re'), data=data, method='REML')
# summary(totalDurGam)
# plot(totalDurGam, seWithMean = TRUE, shift = coef(totalDurGam)[1], shade = TRUE, shade.col = "lightblue", pages=1)
# gam.check(totalDurGam)


# 2. Number of triplets added

# 2.1 First Fix Dur

# baseline
firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
summary(firstFix)

# main model
firstFixInt <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_added + norm_ianum * n_triplets_added + length + frequency + surprisal + (1|participant_id), data = data)
summary(firstFixInt)
# MECO: sig pos effect, no sig int
# Provo: sig pos effect, no sig int

# 2.2. Gaze Duration

# baseline
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
summary(gazeDurBase)

# main model
gazeDurInt <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_added + norm_ianum * n_triplets_added + length + frequency + surprisal + (1|participant_id), data = data)
summary(gazeDurInt)
# MECO: pos effect, negative interaction with position in text
# Provo: sig pos effect of added triplets, sig pos interaction between n of added triplets and norm word position in sentence

# 2.3. Total Reading Time

# baseline
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
summary(totalDur)

# main model
totalDurInt <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_added + norm_ianum * n_triplets_added + length + frequency + surprisal + (1|participant_id), data = data)
summary(totalDurInt)
# MECO: pos effect, negative interactions with word position in sentence and in text
# Provo: pos effect of added triplets, pos interaction between n of added triplets and norm word position in sentence, neg interaction between added triplets and word position in text


# 3. Number of new triplets added

# 3.1 First Fix Dur

# baseline
firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
summary(firstFix)

# main model
firstFixInt <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_new + norm_ianum * n_triplets_new + length + frequency + surprisal + (1|participant_id), data = data)
summary(firstFixInt)
# MECO: pos effect
# Provo: no sig main effect 

# 3.2 Gaze Duration

# baseline
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
summary(gazeDur)

# main model
gazeDurInt <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_new + norm_ianum * n_triplets_new + length + frequency + surprisal + (1|participant_id), data = data)
summary(gazeDurInt)
# MECO: pos effect, neg interaction with word position in sentence and text.
# Provo: sig pos effect, and sig pos int with word position in sentence

# 3.3. Total Reading Time

# baseline
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
summary(totalDur)

# main model
totalDurInt <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_new + norm_ianum * n_triplets_new + length + frequency + surprisal + (1|participant_id), data = data)
summary(totalDurInt)
# MECO: pos effect, neg interaction with word position in sentence and text.
# Provo: sig pos effect, and sig pos int with word position in sentence


# 4. Number of triplets activated

# 4.1 First Fix Dur

# baseline
firstFix <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
summary(firstFix)

# main model
firstFixInt <- lmer(first_fix_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_activated + norm_ianum * n_new_triplets_activated + length + frequency + surprisal + (1|participant_id), data = data)
summary(firstFixInt)
# Provo: neg effect
# MECO: no effect

# 4.2. Gaze Duration

# baseline
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
summary(gazeDur)

# main model
gazeDurInt <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_activated + norm_ianum * n_triplets_activated + length + frequency + surprisal + (1|participant_id), data = data)
summary(gazeDurInt)
# Provo: neg effect
# MECO: no effect

# 4.3. Total Reading Time

# baseline
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + length + frequency + surprisal + norm_ianum + (1|participant_id), data = data)
summary(gazeDur)

# main model
totalDurInt <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * sentnum + norm_word_pos * n_triplets_activated + norm_ianum * n_triplets_activated + length + frequency + surprisal + (1|participant_id), data = data)
summary(gazeDurInt)
# Provo: neg effect
# MECO: no effect

# 5. sanity checks
# AL: strong positive correlation with word pos in text, but weak positive correlation with word pos in sentence.
# co-relation between n of triplets and word position in text
cor.test(data$n_triplets, data$ianum)
# co-relation between n of triplets and word position in sentence
cor.test(data$n_triplets, data$norm_word_pos)
# co-relation between n of triplets and sum scores
cor.test(data$n_triplets, data$sum_scores)
# correlation between n_triplets and n_triplets_added
cor.test(data$n_triplets, data$n_triplets_added)
# correlation between n_triplets_added and surprisal
cor.test(data$n_triplets_added, data$surprisal)

# 6. isolate last sentence and add interaction between n_triplets and norm_word_pos
# AL: only in gaze dur, sig neg interaction between n of triplets and word position in sentence (less acceleration along sentence with more triplets)
data_last_sentence <- data[data$norm_sentnum == 1,]
gazeDur <- lmer(gaze_dur ~ norm_word_pos * sent_length + norm_word_pos * n_triplets + length + frequency + surprisal + ianum + (1|participant_id), data = data_last_sentence)
summary(gazeDur)
totalDur <- lmer(total_dur ~ norm_word_pos * sent_length + norm_word_pos * n_triplets + length + frequency + surprisal + ianum + (1|participant_id), data = data_last_sentence)
summary(totalDur)
