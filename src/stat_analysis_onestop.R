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

model <- 'gpt-4o-mini'
corpus <- 'onestop'
level <- 'article' # paragraph

# read in data
data <- read.csv(glue("output/{model}/{corpus}/{corpus}_articles_eye_mov_plus_triplets_{model}.csv"))

for (run in 1:10) {

  # run_data <- data[data$difficulty_level == 'Adv',]
  run_data <- data %>% filter(`run_id` == run)

  # add triplets n-1 and n+1 variables
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(triplets_added_minus_one = if_else(lag(`n_triplets`, 1) > lag(`n_triplets`, 2), 1, 0)) %>%  ungroup()
  run_data <- run_data %>%  group_by(participant_id, text_id) %>% mutate(triplets_added = if_else(`n_triplets` > lag(`n_triplets`, 1), 1, 0)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(triplets_added_plus_one = if_else(lead(`n_triplets`, 1) > `n_triplets`, 1, 0)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(surprisal_minus_one = lag(`gpt2_surprisal`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(surprisal_plus_one = lead(`gpt2_surprisal`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(frequency_minus_one = lag(`wordfreq_frequency`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(frequency_plus_one = lead(`wordfreq_frequency`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(length_minus_one = lag(`word_length_no_punctuation`, 1)) %>%  ungroup()
  run_data <- run_data %>% group_by(participant_id, text_id) %>% mutate(length_plus_one = lead(`word_length_no_punctuation`, 1)) %>%  ungroup()

  # standardize predictors with z-score
  run_data$norm_word_pos <- scale(run_data$norm_word_pos, center = TRUE, scale = TRUE)
  run_data$abs_word_pos <- scale(run_data$abs_word_pos, center = TRUE, scale = TRUE)
  run_data$sent_length <- scale(run_data$sent_length, center = TRUE, scale = TRUE)
  run_data$length <- scale(run_data$word_length_no_punctuation, center = TRUE, scale = TRUE)
  run_data$frequency <- scale(run_data$wordfreq_frequency, center = TRUE, scale = TRUE)
  run_data$surprisal <- scale(run_data$gpt2_surprisal, center = TRUE, scale = TRUE)
  run_data$ianum <- scale(run_data$ianum, center = TRUE, scale = TRUE)
  run_data$sentnum <- scale(run_data$sent_id, center = TRUE, scale = TRUE)
  run_data$n_triplets <- scale(run_data$n_triplets, center = TRUE, scale = TRUE)
  run_data$norm_ianum <- scale(run_data$norm_ianum, center = TRUE, scale = TRUE)
  run_data$triplets_added <- scale(run_data$triplets_added, center = TRUE, scale = TRUE)
  run_data$triplets_added_minus_one <- scale(run_data$triplets_added_minus_one, center = TRUE, scale = TRUE)
  run_data$triplets_added_plus_one <- scale(run_data$triplets_added_plus_one, center = TRUE, scale = TRUE)
  run_data$surprisal_minus_one <- scale(run_data$surprisal_minus_one, center = TRUE, scale = TRUE)
  run_data$surprisal_plus_one <- scale(run_data$surprisal_plus_one, center = TRUE, scale = TRUE)
  run_data$frequency_minus_one <- scale(run_data$frequency_minus_one, center = TRUE, scale = TRUE)
  run_data$frequency_plus_one <- scale(run_data$frequency_plus_one, center = TRUE, scale = TRUE)
  run_data$length_minus_one <- scale(run_data$length_minus_one, center = TRUE, scale = TRUE)
  run_data$length_plus_one <- scale(run_data$length_plus_one, center = TRUE, scale = TRUE)
  
  # data$n_triplets_added <- scale(data$n_triplets_added, center = TRUE, scale = TRUE)
  # data$n_triplets_new <- scale(data$n_triplets_new, center = TRUE, scale = TRUE)
  # data$n_triplets_activated <- scale(data$n_triplets_activated, center = TRUE, scale = TRUE)
  # data$n_new_triplets_activated <- scale(data$n_new_triplets_activated, center = TRUE, scale = TRUE)

  # convert . into NA in response columns
  run_data$first_fix_dur[run_data$first_fix_dur == "."] <- NA
  run_data$first_fix_dur <- as.numeric(run_data$first_fix_dur)
  run_data$gaze_dur[run_data$gaze_dur == "."] <- NA
  run_data$gaze_dur <- as.numeric(run_data$gaze_dur)
  run_data$total_dur[run_data$total_dur == "."] <- NA
  run_data$total_dur <- as.numeric(run_data$total_dur)
  
  # 1. Number of triplets
  
  # 1.1. First Fix Duration
  
  # baseline
  firstFixBase <- lmer(first_fix_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + (1|article_text_id) + (1|participant_id) + (1|text_id), data = run_data)
  # summary(firstFixBase)
  # Removed abs_word_pos because of perfect co-linearity with norm_word_pos leading to error.
  
  # main model
  firstFix <- lmer(first_fix_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + n_triplets + (1|article_text_id) + (1|participant_id)+ (1|text_id), data = run_data)
  # summary(firstFix)
  
  # save out results
  tidy_model <- tidy(firstFix)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_{run}_{level}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(firstFixBase, firstFix)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_firstFix_{run}_{level}.csv"), row.names = TRUE)
  
  # interaction
  firstFixInt <- lmer(first_fix_dur ~ norm_word_pos * n_triplets + norm_ianum * n_triplets + length + frequency + surprisal + (1|article_text_id) + (1|participant_id)+ (1|text_id), data = run_data)
  nobs(firstFixInt)
  # summary(firstFixInt)
  
  # save model
  saveRDS(firstFixInt, file = glue("analysis/{model}/{corpus}/lmer_firstFixInt_{run}_{level}.rds"))
  
  # save out results
  tidy_model <- tidy(firstFixInt)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFixInt_{run}_{level}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(firstFixBase, firstFixInt)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_firstFixInt_{run}_{level}.csv"), row.names = TRUE)
  
  # 1.2. Gaze Duration
  
  # Baseline
  gazeDurBase <- lmer(gaze_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + (1|article_text_id) + (1|participant_id)+ (1|text_id), data = run_data)
  # summary(gazeDurBase)
  
  # Main model
  gazeDur <- lmer(gaze_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + n_triplets + (1|article_text_id) + (1|participant_id)+ (1|text_id), data = run_data)
  # summary(gazeDur)
  
  # save out results
  tidy_model <- tidy(gazeDur)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_{run}_{level}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(gazeDurBase, gazeDur)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_gazeDur_{run}_{level}.csv"), row.names = TRUE)
  
  # Interaction
  gazeDurInt <- lmer(gaze_dur ~ norm_word_pos * n_triplets + norm_ianum * n_triplets + length + frequency + surprisal + (1|article_text_id) + (1|participant_id)+ (1|text_id), data = run_data)
  # summary(gazeDurInt)
  
  # save model
  saveRDS(gazeDurInt, file = glue("analysis/{model}/{corpus}/lmer_gazeDurInt_{run}_{level}.rds"))
  
  # save out results
  tidy_model <- tidy(gazeDurInt)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDurInt_{run}_{level}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(gazeDurBase, gazeDurInt)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_gazeDurInt_{run}_{level}.csv"), row.names = TRUE)
  
  # 1.3. Total Reading Time
  
  # Baseline
  totalDurBase <- lmer(total_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + (1|article_text_id) + (1|participant_id)+ (1|text_id), data = run_data)
  # summary(totalDurBase)
  
  # Main model
  totalDur <- lmer(total_dur ~ length + frequency + surprisal + norm_ianum + norm_word_pos + n_triplets + (1|article_text_id) + (1|participant_id)+ (1|text_id), data = run_data)
  # summary(totalDur)
  
  # save out results
  tidy_model <- tidy(totalDur)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_{run}_{level}.csv"), row.names = FALSE)
  
  # compare models
  anova_result <- anova(totalDurBase, totalDur)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_totalDur_{run}_{level}.csv"), row.names = TRUE)
  
  # Interaction
  totalDurInt <- lmer(total_dur ~ norm_word_pos * n_triplets + norm_ianum * n_triplets + length + frequency + surprisal + (1|article_text_id) + (1|participant_id)+ (1|text_id), data = run_data)
  # summary(totalDurInt)
  
  # save model
  saveRDS(totalDurInt, file = glue("analysis/{model}/{corpus}/lmer_totalDurInt_{run}_{level}.rds"))
  
  # save out results
  tidy_model <- tidy(totalDurInt)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDurInt_{run}_{level}.csv"), row.names = FALSE)
   
  # compare models
  anova_result <- anova(totalDurBase, totalDurInt)
  write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_totalDurInt_{run}_{level}.csv"), row.names = TRUE)
  
  # # make sure both models have the same rows for anova to work
  # common_rows <- complete.cases(data[, c("triplets_added", "length", "frequency", "surprisal", "norm_ianum", "norm_word_pos")])
  # data <- data[common_rows, ]
  
  # N-1
  firstFixMinusOne <- lmer(first_fix_dur ~ length + frequency + surprisal + norm_word_pos + norm_ianum + n_triplets + triplets_added + triplets_added_minus_one + surprisal_minus_one + frequency_minus_one + length_minus_one + (1|participant_id) + (1|article_text_id)+ (1|text_id), data = run_data)
  saveRDS(firstFixMinusOne, file = glue("analysis/{model}/{corpus}/lmer_firstFix_MinusOne_{run}_{level}.rds"))
  tidy_model <- tidy(firstFixMinusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_MinusOne_{run}_{level}.csv"), row.names = FALSE)
  # anova_result <- anova(firstFix, firstFixMinusOne)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_firstFix_MinusOne_{run}_{level}.csv"), row.names = TRUE)
  
  gazeDurMinusOne <- lmer(gaze_dur ~ length + frequency + surprisal + norm_word_pos + norm_ianum + n_triplets + triplets_added + triplets_added_minus_one + surprisal_minus_one + frequency_minus_one + length_minus_one + (1|participant_id) + (1|article_text_id)+ (1|text_id), data = run_data)
  saveRDS(gazeDurMinusOne, file = glue("analysis/{model}/{corpus}/lmer_gazeDur_MinusOne_{run}_{level}.rds"))
  tidy_model <- tidy(gazeDurMinusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_MinusOne_{run}_{level}.csv"), row.names = FALSE)
  # anova_result <- anova(gazeDur, gazeDurMinusOne)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_gazeDur_MinusOne_{run}_{level}.csv"), row.names = TRUE)
  
  totalDurMinusOne <- lmer(total_dur ~ length + frequency + surprisal + norm_word_pos + norm_ianum + n_triplets + triplets_added + triplets_added_minus_one + surprisal_minus_one + frequency_minus_one + length_minus_one + (1|participant_id) + (1|article_text_id)+ (1|text_id), data = run_data)
  saveRDS(totalDurMinusOne, file = glue("analysis/{model}/{corpus}/lmer_totalDur_MinusOne_{run}_{level}.rds"))
  tidy_model <- tidy(totalDurMinusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_MinusOne_{run}_{level}.csv"), row.names = FALSE)
  # anova_result <- anova(totalDur, totalDurMinusOne)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_totalDur_MinusOne_{run}_{level}.csv"), row.names = TRUE)
  
  # N+1
  firstFixPlusOne <- lmer(first_fix_dur ~ length + frequency + surprisal + norm_word_pos + norm_ianum + n_triplets + triplets_added + triplets_added_plus_one + surprisal_plus_one + frequency_plus_one + length_plus_one + (1|participant_id) + (1|article_text_id)+ (1|text_id), data = run_data)
  saveRDS(firstFixPlusOne, file = glue("analysis/{model}/{corpus}/lmer_firstFix_PlusOne_{run}_{level}.rds"))
  tidy_model <- tidy(firstFixPlusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_firstFix_PlusOne_{run}_{level}.csv"), row.names = FALSE)
  # anova_result <- anova(firstFix, firstFixPlusOne)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_firstFix_PlusOne_{run}_{level}.csv"), row.names = TRUE)
  
  gazeDurPlusOne <- lmer(gaze_dur ~ length + frequency + surprisal + norm_word_pos + norm_ianum + n_triplets + triplets_added + triplets_added_plus_one + surprisal_plus_one + frequency_plus_one + length_plus_one + (1|participant_id) + (1|article_text_id)+ (1|text_id), data = run_data)
  saveRDS(gazeDurPlusOne, file = glue("analysis/{model}/{corpus}/lmer_gazeDur_PlusOne_{run}_{level}.rds"))
  tidy_model <- tidy(gazeDurPlusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_gazeDur_PlusOne_{run}_{level}.csv"), row.names = FALSE)
  # anova_result <- anova(gazeDur, gazeDurPlusOne)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_gazeDur_PlusOne_{run}_{level}.csv"), row.names = TRUE)
  
  totalDurPlusOne <- lmer(total_dur ~ length + frequency + surprisal + norm_word_pos + norm_ianum + n_triplets + triplets_added + triplets_added_plus_one + surprisal_plus_one + frequency_plus_one + length_plus_one + (1|participant_id) + (1|article_text_id)+ (1|text_id), data = run_data)
  saveRDS(totalDurPlusOne, file = glue("analysis/{model}/{corpus}/lmer_totalDur_PlusOne_{run}_{level}.rds"))
  tidy_model <- tidy(totalDurPlusOne)
  write.csv(tidy_model, glue("analysis/{model}/{corpus}/lmer_totalDur_PlusOne_{run}_{level}.csv"), row.names = FALSE)
  # anova_result <- anova(totalDur, totalDurPlusOne)
  # write.csv(anova_result, glue("analysis/{model}/{corpus}/anova_totalDur_PlusOne_{run}_{level}.csv"), row.names = TRUE)
}

# sanity checks
# co-relation between n of triplets and word position in text
cor.test(data$n_triplets, data$ianum)
# co-relation between n of triplets and word position in sentence
cor.test(data$n_triplets, data$norm_word_pos)
# co-relation between n of triplets and sum scores
cor.test(data$n_triplets, data$sum_scores)
# correlation between n of triplets and surprisal
cor.test(data$n_triplets_added, data$surprisal)
