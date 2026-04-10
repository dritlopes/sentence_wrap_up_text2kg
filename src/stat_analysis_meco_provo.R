# install.packages("broom.mixed")
library(lme4)
library(readr)
library(ggplot2)
library(dplyr)
library(lmerTest)
library(mgcv)
library(glue)
library(broom.mixed)
library(ggeffects)

# clear existing workspace objects 
rm(list = ls())
# set working directory to where the data file is located & results should be saved
setwd("/Users/adriellilopes/PycharmProjects/Text2KG/data/")

pre_process <- function(data, corpus){
  if (corpus == 'provo'){
    data$total_dur[data$total_dur == 0] <- NA
    data$total_dur <- as.numeric(data$total_dur)}
  # add triplets n-1 and n+1 variables
  data <- data %>% group_by(participant_id, text_id) %>% mutate(first_fix_dur_minus_one = lag(`first_fix_dur`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(first_fix_dur_plus_one = lead(`first_fix_dur`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(gaze_dur_minus_one = lag(`gaze_dur`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(gaze_dur_plus_one = lead(`gaze_dur`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(total_dur_minus_one = lag(`total_dur`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(total_dur_plus_one = lead(`total_dur`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(n_new_triplets_minus_one = lag(`n_new_triplets`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(n_new_triplets_plus_one = lead(`n_new_triplets`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(triplet_added_minus_one = lag(`triplet_added`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(triplet_added_plus_one = lead(`triplet_added`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(surprisal_minus_one = lag(`surprisal`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(surprisal_plus_one = lead(`surprisal`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(frequency_minus_one = lag(`frequency`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(frequency_plus_one = lead(`frequency`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(length_minus_one = lag(`length`, 1)) %>%  ungroup()
  data <- data %>% group_by(participant_id, text_id) %>% mutate(length_plus_one = lead(`length`, 1)) %>%  ungroup()
  # standardize predictors with z-score
  # so beta coefficient will mean the amount of increase or decrease of dependent variable associated with 1 standard deviation increase of independent variable
  data$norm_word_pos <- scale(data$norm_word_pos, center = TRUE, scale = TRUE)
  data$length <- scale(data$length, center = TRUE, scale = TRUE)
  data$frequency <- scale(data$frequency, center = TRUE, scale = TRUE)
  data$surprisal <- scale(data$surprisal, center = TRUE, scale = TRUE)
  data$norm_ianum <- scale(data$norm_ianum, center = TRUE, scale = TRUE)
  data$n_new_triplets <- scale(data$n_new_triplets, center = TRUE, scale = TRUE)
  data$n_new_triplets_minus_one <- scale(data$n_new_triplets_minus_one, center = TRUE, scale = TRUE)
  data$n_new_triplets_plus_one <- scale(data$n_new_triplets_plus_one, center = TRUE, scale = TRUE)
  data$surprisal_minus_one <- scale(data$surprisal_minus_one, center = TRUE, scale = TRUE)
  data$surprisal_plus_one <- scale(data$surprisal_plus_one, center = TRUE, scale = TRUE)
  data$frequency_minus_one <- scale(data$frequency_minus_one, center = TRUE, scale = TRUE)
  data$frequency_plus_one <- scale(data$frequency_plus_one, center = TRUE, scale = TRUE)
  data$length_minus_one <- scale(data$length_minus_one, center = TRUE, scale = TRUE)
  data$length_plus_one <- scale(data$length_plus_one, center = TRUE, scale = TRUE)
  
  return(data)
}

# read in data
corpus <- 'provo' # provo or meco
model <- 'gpt-4o-mini'

for (run in 1:10) {
  
  data <- read.csv(glue("output/{model}/{corpus}/{corpus}_eye_mov_plus_triplets_{model}_{run}.csv")) 
  data <- pre_process(data,corpus)
  
  # 1. Triplet activation/addition/formation
  
  # 1.1. First Fix Duration
  
  # baseline model
  firstFixBase <- lmer(first_fix_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + (1|participant_id) + (1|text_id), data = data)
  # summary(firstFixBase)
  
  # main model
  firstFix <- lmer(first_fix_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + triplet_added + (1|participant_id) + (1|text_id), data = data)
  # summary(firstFix)
  
  # save model
  saveRDS(firstFix, file = glue("analysis/{model}/{corpus}/lmer_firstFix_triplet_added_{run}.rds"))
  
  # save out results
  tidy_model <- tidy(firstFix)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_triplet_added_{run}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(firstFixBase, firstFix)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_firstFix_triplet_added_{run}.csv"), row.names = TRUE)
  
  # 1.2. Gaze Duration
  
  # baseline model
  gazeDurBase <- lmer(gaze_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + (1|participant_id) + (1|text_id), data = data)
  # summary(gazeDurBase)
  
  # main model
  gazeDur <- lmer(gaze_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + triplet_added + (1|participant_id) + (1|text_id), data = data)
  # summary(gazeDur)
  
  # save model
  saveRDS(gazeDur, file = glue("analysis/{model}/{corpus}/lmer_gazeDur_triplet_added_{run}.rds"))
  
  # save out results
  tidy_model <- tidy(gazeDur)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_triplet_added_{run}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(gazeDurBase, gazeDur)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_gazeDur_triplet_addded_{run}.csv"), row.names = TRUE)
  
  # 1.3. Total Reading Time
  
  # baseline model
  totalDurBase <- lmer(total_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + (1|participant_id) + (1|text_id), data = data)
  # summary(totalDurBase)
  
  # main model
  totalDur <- lmer(total_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + triplet_added + (1|participant_id) + (1|text_id), data = data)
  # summary(totalDur)
  
  # save model
  saveRDS(totalDur, file = glue("analysis/{model}/{corpus}/lmer_totalDur_triplet_added_{run}.rds"))
  
  # save out results
  tidy_model <- tidy(totalDur)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_triplet_added_{run}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(totalDurBase, totalDur)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_totalDur_triplet_added_{run}.csv"), row.names = TRUE)
  
  # 1.4. N-1
  firstFixMinusOne <- lmer(first_fix_dur_minus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_minus_one + surprisal_minus_one + frequency_minus_one + length_minus_one + (1|participant_id) + (1|text_id), data = data)
  tidy_model <- tidy(firstFixMinusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_MinusOne_triplet_added_{run}.csv"), row.names = FALSE)
  saveRDS(firstFixMinusOne, file = glue("analysis/{model}/{corpus}/lmer_firstFix_MinusOne_triplet_added_{run}.rds"))
  
  gazeDurMinusOne <- lmer(gaze_dur_minus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_minus_one + surprisal_minus_one + frequency_minus_one + length_minus_one + (1|participant_id) + (1|text_id), data = data)
  tidy_model <- tidy(gazeDurMinusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_MinusOne_triplet_added_{run}.csv"), row.names = FALSE)
  saveRDS(gazeDurMinusOne, file = glue("analysis/{model}/{corpus}/lmer_gazeDur_MinusOne_triplet_added_{run}.rds"))

  totalDurMinusOne <- lmer(total_dur_minus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_minus_one + surprisal_minus_one + frequency_minus_one + length_minus_one + (1|participant_id) + (1|text_id), data = data)
  tidy_model <- tidy(totalDurMinusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_MinusOne_triplet_added_{run}.csv"), row.names = FALSE)
  saveRDS(totalDurMinusOne, file = glue("analysis/{model}/{corpus}/lmer_totalDur_MinusOne_triplet_added_{run}.rds"))

  # 1.5. N+1
  firstFixPlusOne <- lmer(first_fix_dur_plus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_plus_one + surprisal_plus_one + frequency_plus_one + length_plus_one + (1|participant_id) + (1|text_id), data = data)
  tidy_model <- tidy(firstFixPlusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_PlusOne_triplet_added_{run}.csv"), row.names = FALSE)
  saveRDS(firstFixPlusOne, file = glue("analysis/{model}/{corpus}/lmer_firstFix_PlusOne_triplet_added_{run}.rds"))
  
  gazeDurPlusOne <- lmer(gaze_dur_plus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_plus_one + surprisal_plus_one + frequency_plus_one + length_plus_one + (1|participant_id) + (1|text_id), data = data)
  # tidy_model <- tidy(gazeDurPlusOne)
  # write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_PlusOne_triplet_added_{run}.csv"), row.names = FALSE)
  # saveRDS(gazeDurPlusOne, file = glue("analysis/{model}/{corpus}/lmer_gazeDur_PlusOne_triplet_added_{run}.rds"))
  
  totalDurPlusOne <- lmer(total_dur_plus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_plus_one + surprisal_plus_one + frequency_plus_one + length_plus_one + (1|participant_id) + (1|text_id), data = data)
  # tidy_model <- tidy(totalDurPlusOne)
  # write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_PlusOne_triplet_added_{run}.csv"), row.names = FALSE)
  # saveRDS(totalDurPlusOne, file = glue("analysis/{model}/{corpus}/lmer_totalDur_PlusOne_triplet_added_{run}.rds"))
}

# Post-hoc analysis - interaction between surprisal and triplet formation
corpus <- 'meco'
model <- 'gpt-4o-mini'
run <- 6
data <- read.csv(glue("output/{model}/{corpus}/{corpus}_eye_mov_plus_triplets_{model}_{run}.csv")) 
data <- pre_process(data,corpus)

# First Fix
# model with interaction
firstFixInt <- lmer(first_fix_dur ~ length + frequency + norm_ianum + norm_word_pos + surprisal*triplet_added + (1|participant_id) + (1|text_id), data = data)
summary(firstFixInt)
# compare models
firstFix <- readRDS(glue("analysis/{model}/{corpus}/lmer_firstFix_triplet_added_{run}.rds"))
anova(firstFix, firstFixInt)

# Gaze Dur
# model with interaction
gazeDurInt <- lmer(gaze_dur ~ length + frequency + norm_ianum + norm_word_pos + triplet_added * surprisal + (1|participant_id) + (1|text_id), data = data)
summary(gazeDurInt)
# compare models
gazeDur <- readRDS(glue("analysis/{model}/{corpus}/lmer_gazeDur_triplet_added_{run}.rds"))
anova(gazeDur, gazeDurInt)

# Total Dur
# model with interaction
totalDurInt <- lmer(total_dur ~ length + frequency + norm_ianum + norm_word_pos + triplet_added * surprisal + (1|participant_id) + (1|text_id), data = data)
summary(totalDurInt)
# compare models
totalDur <- readRDS(glue("analysis/{model}/{corpus}/lmer_totalDur_triplet_added_{run}.rds"))
anova(totalDur, totalDurInt)

# N-1
corpus <- 'provo'
model <- 'gpt-4o-mini'
run <- 4
data <- read.csv(glue("output/{model}/{corpus}/{corpus}_eye_mov_plus_triplets_{model}_{run}.csv")) 
data <- pre_process(data,corpus)
firstFixMinusOneInt <- lmer(first_fix_dur_minus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_minus_one + frequency_minus_one + length_minus_one + (1|participant_id) + (1|text_id), data = data)
summary(firstFixMinusOneInt)
gazeDurMinusOneInt <- lmer(gaze_dur_minus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_minus_one + frequency_minus_one + length_minus_one + (1|participant_id) + (1|text_id), data = data)
summary(gazeDurMinusOneInt)
totalDurMinusOneInt <- lmer(total_dur_minus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_minus_one + frequency_minus_one + length_minus_one + (1|participant_id) + (1|text_id), data = data)
summary(totalDurMinusOneInt)

# N+1
corpus <- 'provo'
model <- 'gpt-4o-mini'
run <- 5
data <- read.csv(glue("output/{model}/{corpus}/{corpus}_eye_mov_plus_triplets_{model}_{run}.csv")) 
data <- pre_process(data,corpus)
firstFixPlusOneInt <- lmer(first_fix_dur_plus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_plus_one + frequency_plus_one + length_plus_one + (1|participant_id) + (1|text_id), data = data)
summary(firstFixPlusOneInt)
gazeDurPlusOneInt <- lmer(gaze_dur_plus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_plus_one + frequency_plus_one + length_plus_one + (1|participant_id) + (1|text_id), data = data)
summary(gazeDurPlusOneInt)
totalDurPlusOneInt <- lmer(total_dur_plus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_plus_one + frequency_plus_one + length_plus_one + (1|participant_id) + (1|text_id), data = data)
summary(totalDurPlusOneInt)