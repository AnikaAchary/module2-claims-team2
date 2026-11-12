## MULTICLASS MODEL (task 4b)

## SETUP ---------------------------------------------------------------------
library(tidyverse)
library(tidymodels)
library(Matrix)
library(sparsesvd)
library(glmnet)

source("scripts/projection-functions.R")
source("scripts/task4/bclass-functions.R")


## DATA ----------------------------------------------------------------------
load("data/claims-raw.RData")
load("data/final-clean.RData")   # cleaned pages from task 4a

train_clean <- train_clean %>%
  left_join(select(claims_raw, .id, mclass), by = ".id")

mlv <- levels(train_clean$mclass)
mlv

nrow(train_clean)                # 2165
sum(is.na(train_clean$mclass))   # 0
table(train_clean$mclass)

# N/A matches between bclass and mclass
table(train_clean$bclass, train_clean$mclass)

# HTML pages only
train_clean %>%
  filter(parsed) %>%
  count(mclass)

tok <- tokenize_pages(train_clean)
n_distinct(tok$.id)
