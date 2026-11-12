## 0. SETUP ------------------------------------------------------------------
library(tidyverse)
library(tidymodels)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(Matrix)
library(sparsesvd)
library(glmnet)
library(e1071) # for the SVM

source('scripts/preprocessing.R')          # starter file, left in place
source('scripts/projection-functions.R')   # in-class projection_fn(), reproject_fn()

# headers version of the TF-IDF table, from Phase 1
load('data/tfidf-headers-test.RData')
tfidf <- tfidf_ph



## MODEL FUNCTIONS -------------------------------------------------------------

panel <- metric_set(sensitivity, specificity, accuracy, roc_auc)

# turn probabilities into labels + metrics
score <- function(truth, p){
  lv <- levels(truth)
  tibble(bclass = truth,
         pred = p,
         bclass.pred = factor(if_else(p > 0.5, lv[2], lv[1]), levels = lv)) %>%
    panel(truth = bclass, estimate = bclass.pred, pred, event_level = 'second')
}

# baseline: PCA + elastic net logistic regression (in-class method)
fit_pcr <- function(tfidf, train_ids, test_ids, prop = 0.7, alpha = 0.3){
  train <- tfidf %>% filter(.id %in% train_ids) %>% arrange(.id)
  test  <- tfidf %>% filter(.id %in% test_ids)  %>% arrange(.id)
  
  proj_out <- projection_fn(.dtm = select(train, -.id, -bclass), .prop = prop)
  x_train  <- as.matrix(proj_out$data)
  x_test   <- as.matrix(reproject_fn(.dtm = select(test, -.id, -bclass), proj_out))
  
  set.seed(102926)
  cvout <- cv.glmnet(x = x_train, y = train$bclass, family = 'binomial', alpha = alpha)
  p <- as.numeric(predict(cvout, s = 'lambda.min', newx = x_test, type = 'response'))
  score(test$bclass, p)
}

# challenger: glmnet directly on TF-IDF words (no PCA)
fit_enet <- function(tfidf, train_ids, test_ids, alpha = 1){
  train <- tfidf %>% filter(.id %in% train_ids) %>% arrange(.id)
  test  <- tfidf %>% filter(.id %in% test_ids)  %>% arrange(.id)
  
  x_train <- Matrix(as.matrix(select(train, -.id, -bclass)), sparse = TRUE)
  x_test  <- Matrix(as.matrix(select(test,  -.id, -bclass)), sparse = TRUE)
  
  set.seed(102926)
  cvout <- cv.glmnet(x = x_train, y = train$bclass, family = 'binomial', alpha = alpha)
  p <- as.numeric(predict(cvout, s = 'lambda.min', newx = x_test, type = 'response'))
  score(test$bclass, p)
}

# run a fit function over 10 random splits (same splits for every model)
evaluate <- function(fit_fun, tfidf, seeds = 1:10, ...){
  map_dfr(seeds, function(s){
    set.seed(s)
    sp <- tfidf %>%
      arrange(.id) %>%
      select(.id, bclass) %>%
      initial_split(prop = 0.8, strata = bclass)
    fit_fun(tfidf, training(sp)$.id, testing(sp)$.id, ...) %>%
      mutate(seed = s)
  })
}


## 4-5. BASELINE + CANDIDATES (same 10 splits) ----------------------------------

res_baseline <- evaluate(fit_pcr,  tfidf)              %>% mutate(model = 'PCA + logistic (baseline)')
res_lasso    <- evaluate(fit_enet, tfidf, alpha = 1)   %>% mutate(model = 'glmnet lasso')
res_enet     <- evaluate(fit_enet, tfidf, alpha = 0.5) %>% mutate(model = 'glmnet elastic net')
res_ridge    <- evaluate(fit_enet, tfidf, alpha = 0)   %>% mutate(model = 'glmnet ridge')

results_p2 <- bind_rows(res_baseline, res_lasso, res_enet, res_ridge)
save(results_p2, file = 'results/binary-comparison.RData')

results_p2 %>%
  group_by(model, .metric) %>%
  summarise(mean = mean(.estimate), sd = sd(.estimate), .groups = 'drop') %>%
  pivot_wider(names_from = .metric, values_from = c(mean, sd))

## RESULT (10 splits): all linear models ~80% accuracy
##   baseline PCA+logistic 0.800 (AUC 0.865) | lasso 0.805 (0.860)
##   elastic net 0.803 (0.860) | ridge 0.794 (0.853, sens 0.838 / spec 0.746)
## Differences are within split noise (~1.5 pts). No linear model beats the baseline.


## =====================================================================
## PROGRESS SO FAR (task 4a, binary model)

## PHASE 1: headers (done, see scripts/task4/headers-test.R)
##   - Cleaned pages two ways: paragraphs only vs paragraphs + headers (h1-h6)
##   - Headers cut blank pages from 168 to 41 (median words 223 -> 281)
##   - PCA + logistic regression over 10 splits:
##       accuracy 0.793 vs 0.791 (no change), AUC 0.847 vs 0.864 (headers win 8/10)
##   - Decision: keep headers
##
## PHASE 2: model comparison (in progress, this script)
##   - Same 10 splits for every model, headers TF-IDF (words on 5+ pages)
##   - Baseline PCA + logistic: accuracy 0.800, AUC 0.865
##   - glmnet on TF-IDF: lasso 0.805, elastic net 0.803, ridge 0.794
##   - All within split noise (~1.5 pts); no model beats the baseline yet
##   - Neural net skipped (Keras setup issues)
##   - Next: bigrams + lasso, then optional non-linear SVM
## =====================================================================


## BIGRAMS -----------------------------------------------------------------

# cleaned text with headers, from Phase 1 (gives claims_p and claims_ph)
load('data/claims-clean-headers-test.RData')

bigram_fn <- function(.df, min_docs = 10){
  stops <- str_remove_all(stop_words$word, '[[:punct:]]')
  
  # pairs of neighboring words, stop words and short words removed, then lemmatized
  bigrams <- .df %>%
    select(.id, bclass, text_clean) %>%
    unnest_tokens(output = bigram, input = text_clean, token = 'ngrams', n = 2) %>%
    filter(!is.na(bigram)) %>%
    separate(bigram, into = c('w1', 'w2'), sep = ' ') %>%
    filter(!w1 %in% stops, !w2 %in% stops,
           str_length(w1) > 2, str_length(w2) > 2) %>%
    mutate(token.lem = paste('bg', lemmatize_words(w1), lemmatize_words(w2), sep = '_')) %>%
    count(.id, bclass, token.lem, name = 'n')
  
  # keep bigrams that appear on at least min_docs pages
  keep <- bigrams %>%
    count(token.lem, name = 'docs') %>%
    filter(docs >= min_docs) %>%
    pull(token.lem)
  
  # TF-IDF, one row per page, one column per bigram
  tfidf <- bigrams %>%
    filter(token.lem %in% keep) %>%
    bind_tf_idf(term = token.lem, document = .id, n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  
  # pages with no kept bigrams become rows of zeros
  blanks <- .df %>%
    filter(!.id %in% tfidf$.id) %>%
    select(.id, bclass)
  
  bind_rows(tfidf, blanks) %>%
    mutate(across(-c(.id, bclass), ~ replace_na(.x, 0)))
}

bigrams_ph <- bigram_fn(claims_ph)
save(bigrams_ph, file = 'data/tfidf-bigrams.RData')

# words + bigrams side by side, one row per page
tfidf_wb <- tfidf %>%
  inner_join(select(bigrams_ph, -bclass), by = '.id')

dim(bigrams_ph)   # expect 2140 rows
dim(tfidf_wb)     # expect 2140 rows, words + bigram columns


res_lasso_bg <- evaluate(fit_enet, tfidf_wb, alpha = 1) %>%
  mutate(model = 'glmnet lasso + bigrams')

results_p2 <- bind_rows(results_p2, res_lasso_bg)
save(results_p2, file = 'results/binary-comparison.RData')

results_p2 %>%
  group_by(model, .metric) %>%
  summarise(mean = mean(.estimate), sd = sd(.estimate), .groups = 'drop') %>%
  pivot_wider(names_from = .metric, values_from = c(mean, sd))

## RESULT (10 splits): bigrams add nothing
##   lasso words only 0.805 (AUC 0.860) | lasso words + 1,562 bigrams 0.805 (AUC 0.859)
## Bigrams that appear on 10+ pages carry no extra signal beyond single words.



# install.packages('e1071')


# SVM with a non-linear (RBF) kernel, fit on the PCA components
fit_svm <- function(tfidf, train_ids, test_ids, prop = 0.7, cost = 1){
  train <- tfidf %>% filter(.id %in% train_ids) %>% arrange(.id)
  test  <- tfidf %>% filter(.id %in% test_ids)  %>% arrange(.id)
  
  # same projection as the baseline
  proj_out <- projection_fn(.dtm = select(train, -.id, -bclass), .prop = prop)
  x_train  <- as.matrix(proj_out$data)
  x_test   <- as.matrix(reproject_fn(.dtm = select(test, -.id, -bclass), proj_out))
  
  # probability = TRUE so we get P(Relevant) for AUC
  set.seed(102926)
  fit <- svm(x = x_train, y = train$bclass, kernel = 'radial',
             cost = cost, probability = TRUE)
  
  pr <- predict(fit, x_test, probability = TRUE)
  p  <- attr(pr, 'probabilities')[, levels(train$bclass)[2]]
  score(test$bclass, p)
}


res_svm1  <- evaluate(fit_svm, tfidf, cost = 1)  %>% mutate(model = 'SVM radial (cost 1)')
res_svm10 <- evaluate(fit_svm, tfidf, cost = 10) %>% mutate(model = 'SVM radial (cost 10)')

results_p2 <- bind_rows(results_p2, res_svm1, res_svm10)
save(results_p2, file = 'results/binary-comparison.RData')

results_p2 %>%
  group_by(model, .metric) %>%
  summarise(mean = mean(.estimate), sd = sd(.estimate), .groups = 'drop') %>%
  pivot_wider(names_from = .metric, values_from = c(mean, sd))




## RESULT (10 splits): SVM adds nothing
##   SVM radial cost 1: 0.794 (AUC 0.862) | cost 10: 0.801 (AUC 0.862)
##
## PHASE 2 DECISION: glmnet lasso on word TF-IDF (headers included)
##   accuracy 0.805, AUC 0.860, sens 0.798, spec 0.813
##   All 7 models fell within ~1 pt of 80%; lasso ties for best accuracy,
##   is balanced across classes, and is the simplest to deploy (no PCA).


## FOR DELIVERABLE 2:
## Seven models (PCA logistic regression, lasso, elastic net, ridge,
## lasso with bigrams, and two RBF SVMs) all reached about 80% accuracy
## over 10 splits; lasso logistic regression on word TF-IDF was chosen
## (80.5% accuracy, 0.860 AUC).


