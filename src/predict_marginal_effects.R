library(lme4)
library(readr)
library(ggplot2)
library(dplyr)
library(lmerTest)
library(mgcv)
library(glue)
library(broom.mixed)
library(ggeffects)

rm(list = ls())
setwd("/Users/adriellilopes/PycharmProjects/Text2KG/data/")

model_name <- 'gpt-4o-mini'
eye_mov_measure <- list('firstFix','gazeDur','totalDur')
term <- list('length_minus_one','frequency_minus_one','surprisal_minus_one','norm_ianum','norm_word_pos')
model_type <- '_MinusOne'
# median models based on mean beta coefficient of number of triplets over eye movement measures (computed in Python)
provo_median_model <- 4
meco_median_model <- 10
onestop_median_model <- 7

for (measure in eye_mov_measure){
  for(predictor in term){
    provo_model <-readRDS(glue('analysis/{model_name}/provo/lmer_{measure}{model_type}_triplet_added_{provo_median_model}.rds'))
    provo_effects <- ggpredict(provo_model, terms = predictor)
    write.csv(provo_effects, glue("analysis/{model_name}/provo/predicted_marginal_{predictor}_{measure}{model_type}.csv"), row.names = TRUE)
    meco_model <- readRDS(glue('analysis/{model_name}/meco/lmer_{measure}{model_type}_triplet_added_{meco_median_model}.rds')) # marginal effects of fixed factor
    meco_effects <- ggpredict(meco_model, terms = predictor)
    write.csv(meco_effects, glue("analysis/{model_name}/meco/predicted_marginal_{predictor}_{measure}{model_type}.csv"), row.names = TRUE)
    onestop_model <-readRDS(glue('analysis/{model_name}/onestop/lmer_{measure}{model_type}_triplet_added_{onestop_median_model}_paragraph.rds'))
    onestop_effects <- ggpredict(onestop_model, terms = predictor)
    write.csv(onestop_effects, glue("analysis/{model_name}/onestop/predicted_marginal_{predictor}_{measure}{model_type}.csv"), row.names = TRUE)
  }
}