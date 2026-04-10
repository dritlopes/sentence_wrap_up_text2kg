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
setwd(glue("/Users/adriellilopes/PycharmProjects/Text2KG/data/"))

pre_process <- function(run_data){
  
  # add triplets n-1 and n+1 variables
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(first_fix_dur_minus_one = lag(`first_fix_dur`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(first_fix_dur_plus_one = lead(`first_fix_dur`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(gaze_dur_minus_one = lag(`gaze_dur`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(gaze_dur_plus_one = lead(`gaze_dur`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(total_dur_minus_one = lag(`total_dur`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(total_dur_plus_one = lead(`total_dur`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(n_new_triplets_minus_one = lag(`n_new_triplets`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(n_new_triplets_plus_one = lead(`n_new_triplets`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(triplet_added_minus_one = lag(`triplet_added`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(triplet_added_plus_one = lead(`triplet_added`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(surprisal_minus_one = lag(`gpt2_surprisal`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(surprisal_plus_one = lead(`gpt2_surprisal`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(frequency_minus_one = lag(`wordfreq_frequency`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(frequency_plus_one = lead(`wordfreq_frequency`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(length_minus_one = lag(`word_length_no_punctuation`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(length_plus_one = lead(`word_length_no_punctuation`, 1)) %>%  ungroup()
  
  # standardize predictors with z-score
  run_data$norm_word_pos <- scale(run_data$norm_word_pos, center = TRUE, scale = TRUE)
  run_data$length <- scale(run_data$word_length_no_punctuation, center = TRUE, scale = TRUE)
  run_data$frequency <- scale(run_data$wordfreq_frequency, center = TRUE, scale = TRUE)
  run_data$surprisal <- scale(run_data$gpt2_surprisal, center = TRUE, scale = TRUE)
  run_data$norm_ianum <- scale(run_data$norm_ianum, center = TRUE, scale = TRUE)
  run_data$n_new_triplets <- scale(run_data$n_new_triplets, center = TRUE, scale = TRUE)
  run_data$n_new_triplets_minus_one <- scale(run_data$n_new_triplets_minus_one, center = TRUE, scale = TRUE)
  run_data$n_new_triplets_plus_one <- scale(run_data$n_new_triplets_plus_one, center = TRUE, scale = TRUE)
  run_data$surprisal_minus_one <- scale(run_data$surprisal_minus_one, center = TRUE, scale = TRUE)
  run_data$surprisal_plus_one <- scale(run_data$surprisal_plus_one, center = TRUE, scale = TRUE)
  run_data$frequency_minus_one <- scale(run_data$frequency_minus_one, center = TRUE, scale = TRUE)
  run_data$frequency_plus_one <- scale(run_data$frequency_plus_one, center = TRUE, scale = TRUE)
  run_data$length_minus_one <- scale(run_data$length_minus_one, center = TRUE, scale = TRUE)
  run_data$length_plus_one <- scale(run_data$length_plus_one, center = TRUE, scale = TRUE)
  
  # convert . into NA in response columns
  run_data$first_fix_dur[run_data$first_fix_dur == "."] <- NA
  run_data$first_fix_dur <- as.numeric(run_data$first_fix_dur)
  run_data$first_fix_dur_minus_one[run_data$first_fix_dur_minus_one == "."] <- NA
  run_data$first_fix_dur_minus_one <- as.numeric(run_data$first_fix_dur_minus_one)
  run_data$first_fix_dur_plus_one[run_data$first_fix_dur_plus_one == "."] <- NA
  run_data$first_fix_dur_plus_one <- as.numeric(run_data$first_fix_dur_plus_one)
  
  run_data$gaze_dur[run_data$gaze_dur == "."] <- NA
  run_data$gaze_dur <- as.numeric(run_data$gaze_dur)
  run_data$gaze_dur_minus_one[run_data$gaze_dur_minus_one == "."] <- NA
  run_data$gaze_dur_minus_one <- as.numeric(run_data$gaze_dur_minus_one)
  run_data$gaze_dur_plus_one[run_data$gaze_dur_plus_one == "."] <- NA
  run_data$gaze_dur_plus_one <- as.numeric(run_data$gaze_dur_plus_one)
  
  run_data$total_dur[run_data$total_dur == 0] <- NA
  run_data$total_dur <- as.numeric(run_data$total_dur)
  run_data$total_dur_minus_one[run_data$total_dur_minus_one == 0] <- NA
  run_data$total_dur_minus_one <- as.numeric(run_data$total_dur_minus_one)
  run_data$total_dur_plus_one[run_data$total_dur_plus_one == 0] <- NA
  run_data$total_dur_plus_one <- as.numeric(run_data$total_dur_plus_one)
  
  return(run_data)
}

model <- 'gpt-4o-mini'
corpus <- 'onestop'
level <- 'paragraph' # paragraph, article

for (run in 1:10) {

  run_data <- read.csv(glue("output/{model}/{corpus}/{corpus}_eye_mov_plus_triplets_{model}_{run}.csv"))
  run_data <- pre_process(run_data)
  
  # 1. Triplet activation/addition/formation
  
  # 1.1. First Fix Duration
  
  # baseline model
  firstFixBase <- lmer(first_fix_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  # summary(firstFixBase)
  
  # main model
  firstFix <- lmer(first_fix_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + triplet_added + (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  # summary(firstFix)
  
  # save model
  saveRDS(firstFix, file = glue("analysis/{model}/{corpus}/lmer_firstFix_triplet_added_{run}_{level}.rds"))
  
  # save out results
  tidy_model <- tidy(firstFix)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_triplet_added_{run}_{level}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(firstFixBase, firstFix)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_firstFix_triplet_added_{run}_{level}.csv"), row.names = TRUE)
  
  # 1.2. Gaze Duration
  
  # baseline model
  gazeDurBase <- lmer(gaze_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos+ (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  # summary(gazeDurBase)
  
  # main model
  gazeDur <- lmer(gaze_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + triplet_added+ (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  # summary(gazeDur)
  
  # save model
  saveRDS(gazeDur, file = glue("analysis/{model}/{corpus}/lmer_gazeDur_triplet_added_{run}_{level}.rds"))
  
  # save out results
  tidy_model <- tidy(gazeDur)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_triplet_added_{run}_{level}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(gazeDurBase, gazeDur)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_gazeDur_triplet_addded_{run}_{level}.csv"), row.names = TRUE)
  
  # 1.3. Total Reading Time
  
  # baseline model
  totalDurBase <- lmer(total_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos+ (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  # summary(totalDurBase)
  
  # main model
  totalDur <- lmer(total_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + triplet_added + (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  # summary(totalDur)
  
  # save model
  saveRDS(totalDur, file = glue("analysis/{model}/{corpus}/lmer_totalDur_triplet_added_{run}_{level}.rds"))
  
  # save out results
  tidy_model <- tidy(totalDur)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_triplet_added_{run}_{level}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(totalDurBase, totalDur)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_totalDur_triplet_added_{run}_{level}.csv"), row.names = TRUE)
  
  # 1.4. N-1
  firstFixMinusOne <- lmer(first_fix_dur_minus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_minus_one + surprisal_minus_one + frequency_minus_one + length_minus_one+ (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  tidy_model <- tidy(firstFixMinusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_MinusOne_triplet_added_{run}_{level}.csv"), row.names = FALSE)
  saveRDS(firstFixMinusOne, file = glue("analysis/{model}/{corpus}/lmer_firstFix_MinusOne_triplet_added_{run}_{level}.rds"))
  
  gazeDurMinusOne <- lmer(gaze_dur_minus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_minus_one + surprisal_minus_one + frequency_minus_one + length_minus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  tidy_model <- tidy(gazeDurMinusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_MinusOne_triplet_added_{run}_{level}.csv"), row.names = FALSE)
  saveRDS(gazeDurMinusOne, file = glue("analysis/{model}/{corpus}/lmer_gazeDur_MinusOne_triplet_added_{run}_{level}.rds"))
  
  totalDurMinusOne <- lmer(total_dur_minus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_minus_one + surprisal_minus_one + frequency_minus_one + length_minus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  tidy_model <- tidy(totalDurMinusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_MinusOne_triplet_added_{run}_{level}.csv"), row.names = FALSE)
  saveRDS(totalDurMinusOne, file = glue("analysis/{model}/{corpus}/lmer_totalDur_MinusOne_triplet_added_{run}_{level}.rds"))
  
  # 1.5. N+1
  firstFixPlusOne <- lmer(first_fix_dur_plus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_plus_one + surprisal_plus_one + frequency_plus_one + length_plus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  tidy_model <- tidy(firstFixPlusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_PlusOne_triplet_added_{run}_{level}.csv"), row.names = FALSE)
  saveRDS(firstFixPlusOne, file = glue("analysis/{model}/{corpus}/lmer_firstFix_PlusOne_triplet_added_{run}_{level}.rds"))
  
  gazeDurPlusOne <- lmer(gaze_dur_plus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_plus_one + surprisal_plus_one + frequency_plus_one + length_plus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  tidy_model <- tidy(gazeDurPlusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_PlusOne_triplet_added_{run}_{level}.csv"), row.names = FALSE)
  saveRDS(gazeDurPlusOne, file = glue("analysis/{model}/{corpus}/lmer_gazeDur_PlusOne_triplet_added_{run}_{level}.rds"))
  
  totalDurPlusOne <- lmer(total_dur_plus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added + triplet_added_plus_one + surprisal_plus_one + frequency_plus_one + length_plus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  tidy_model <- tidy(totalDurPlusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_PlusOne_triplet_added_{run}_{level}.csv"), row.names = FALSE)
  saveRDS(totalDurPlusOne, file = glue("analysis/{model}/{corpus}/lmer_totalDur_PlusOne_triplet_added_{run}_{level}.rds"))
  
}

# Post-hoc analysis - interaction between surprisal and triplet formation
corpus <- 'onestop'
model <- 'gpt-4o-mini'
run <- 6
data <- read.csv(glue("output/{model}/{corpus}/{corpus}_eye_mov_plus_triplets_{model}_{run}.csv"))
data <- pre_process(data)

# First Fix
firstFixInt <- lmer(first_fix_dur ~ length + frequency + norm_ianum + norm_word_pos + triplet_added * surprisal + (1|participant_id) + (1|article_text_id) + (1|text_id), data = data)
summary(firstFixInt)

# Gaze Dur
gazeDurInt <- lmer(gaze_dur ~ length + frequency + norm_ianum + norm_word_pos + triplet_added * surprisal + (1|participant_id) + (1|article_text_id) + (1|text_id), data = data)
summary(gazeDurInt)

# Total Dur
totalDurInt <- lmer(total_dur ~ length + frequency + norm_ianum + norm_word_pos + triplet_added * surprisal + (1|participant_id) + (1|article_text_id) + (1|text_id), data = data)
summary(totalDurInt)

# N-1
run <- 7
data <- read.csv(glue("output/{model}/{corpus}/{corpus}_eye_mov_plus_triplets_{model}_{run}.csv"))
data <- pre_process(data)
firstFixMinusOneInt <- lmer(first_fix_dur_minus_one ~ length + frequency + surprisal + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + triplet_added_minus_one + frequency_minus_one + length_minus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = data)
summary(firstFixMinusOneInt)
gazeDurMinusOneInt <- lmer(gaze_dur_minus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_minus_one + frequency_minus_one + length_minus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = data)
summary(gazeDurMinusOneInt)
totalDurMinusOneInt <- lmer(total_dur_minus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_minus_one + frequency_minus_one + length_minus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = data)
summary(totalDurMinusOneInt)

# N+1
run <- 1
data <- read.csv(glue("output/{model}/{corpus}/{corpus}_eye_mov_plus_triplets_{model}_{run}.csv")) 
data <- pre_process(data)
firstFixPlusOneInt <- lmer(first_fix_dur_plus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_plus_one + frequency_plus_one + length_plus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = data)
summary(firstFixPlusOneInt)
gazeDurPlusOneInt <- lmer(gaze_dur_plus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_plus_one + frequency_plus_one + length_plus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = data)
summary(gazeDurPlusOneInt)
totalDurPlusOneInt <- lmer(total_dur_plus_one ~ length + frequency + norm_word_pos + norm_ianum + triplet_added * surprisal_minus_one + surprisal + triplet_added_plus_one + frequency_plus_one + length_plus_one + (1|article_text_id) + (1|participant_id) + (1|text_id), data = data)
summary(totalDurPlusOneInt)


