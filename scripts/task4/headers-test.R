## Phase 1: should we include headers?

# packages
library(tidyverse)
library(tidymodels)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)
library(Matrix)
library(sparsesvd)
library(glmnet)

# shared cleaning functions from the repo
source('scripts/preprocessing.R')

# save a local copy of the in-class projection functions (only needs to run once)
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
download.file(paste0(url, 'projection-functions.R'),
              destfile = 'scripts/projection-functions.R')
source('scripts/projection-functions.R')

# load the labeled pages
load('data/claims-raw.RData')


dim(claims_raw)            # expect 2165 9
table(claims_raw$bclass)   # expect 1032 N/A, 1133 Relevant
exists('projection_fn')    # expect TRUE
exists('parse_data')       # expect TRUE

## cleaning with a choice of tags --------------------------------------


parse_fn2 <- function(.html, .css = 'p'){
  read_html(.html) %>%
    html_elements(.css) %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', '[[:punct:]]', 'nbsp',
                            '[[:digit:]]', '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# same as parse_data(), passing .css through
parse_data2 <- function(.df, .css = 'p'){
  .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn2(text_tmp, .css)) %>%
    ungroup()
}

# clean both ways 
claims_p  <- parse_data2(claims_raw, .css = 'p')
claims_ph <- parse_data2(claims_raw, .css = 'p, h1, h2, h3, h4, h5, h6')

# save
save(claims_p, claims_ph, file = 'data/claims-clean-headers-test.RData')

nrow(claims_p); nrow(claims_ph)                  # expect 2140 each
sum(str_trim(claims_p$text_clean) == '')         # blank pages, paragraphs only (expect ~168)
sum(str_trim(claims_ph$text_clean) == '')        # blank pages with headers (should be lower)
median(str_count(claims_p$text_clean, '\\S+'))   # median words per page
median(str_count(claims_ph$text_clean, '\\S+'))









## TF-IDF ---------------------------------------------------------------

nlp_fn2 <- function(.df, min_docs = 5){
  # words -> lemmas -> counts per page (same steps as nlp_fn)
  tokens <- .df %>%
    select(.id, bclass, text_clean) %>%
    unnest_tokens(output = token, input = text_clean, token = 'words',
                  stopwords = str_remove_all(stop_words$word, '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n')
  
  # keep lemmas that appear on at least min_docs pages
  keep <- tokens %>%
    count(token.lem, name = 'docs') %>%
    filter(docs >= min_docs) %>%
    pull(token.lem)
  
  # TF-IDF, one row per page, one column per lemma
  tfidf <- tokens %>%
    filter(token.lem %in% keep) %>%
    bind_tf_idf(term = token.lem, document = .id, n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  
  # add back blank pages as rows of zeros
  blanks <- .df %>%
    filter(!.id %in% tfidf$.id) %>%
    select(.id, bclass)
  
  bind_rows(tfidf, blanks) %>%
    mutate(across(-c(.id, bclass), ~ replace_na(.x, 0)))
}

tfidf_p  <- nlp_fn2(claims_p)
tfidf_ph <- nlp_fn2(claims_ph)

save(tfidf_p, tfidf_ph, file = 'data/tfidf-headers-test.RData')



dim(tfidf_p)                            # expect 2140 rows
dim(tfidf_ph)                           # expect 2140 rows, a few more columns
setequal(tfidf_p$.id, tfidf_ph$.id)     # expect TRUE



## split -----------------------------------------------------------------

# choose train/test pages once, by .id (same seed as the class activity)
set.seed(102926)
split_ids <- tfidf_p %>%
  select(.id, bclass) %>%
  initial_split(prop = 0.8, strata = bclass)

train_ids <- training(split_ids)$.id
test_ids  <- testing(split_ids)$.id



## PC logistic regression  -------------------------------

panel <- metric_set(sensitivity, specificity, accuracy, roc_auc)

fit_pcr <- function(tfidf, train_ids, test_ids, prop = 0.7, alpha = 0.3){
  # same pages, same order, for every version
  train <- tfidf %>% filter(.id %in% train_ids) %>% arrange(.id)
  test  <- tfidf %>% filter(.id %in% test_ids)  %>% arrange(.id)
  
  # project onto principal components found from training pages only
  proj_out <- projection_fn(.dtm = select(train, -.id, -bclass), .prop = prop)
  x_train  <- as.matrix(proj_out$data)
  x_test   <- as.matrix(reproject_fn(.dtm = select(test, -.id, -bclass), proj_out))
  
  # elastic net logistic regression, lambda chosen by cross-validation
  set.seed(102926)
  cvout <- cv.glmnet(x = x_train, y = train$bclass,
                     family = 'binomial', alpha = alpha)
  
  # predicted probabilities and labels for the test pages
  p  <- as.numeric(predict(cvout, s = 'lambda.min', newx = x_test, type = 'response'))
  lv <- levels(train$bclass)
  
  tibble(bclass = test$bclass,
         pred = p,
         bclass.pred = factor(if_else(p > 0.5, lv[2], lv[1]), levels = lv)) %>%
    panel(truth = bclass, estimate = bclass.pred, pred, event_level = 'second') %>%
    mutate(n_pc = proj_out$n_pc)
}

results <- bind_rows(
  fit_pcr(tfidf_p,  train_ids, test_ids) %>% mutate(version = 'paragraphs'),
  fit_pcr(tfidf_ph, train_ids, test_ids) %>% mutate(version = 'paragraphs + headers')
)

results %>%
  select(version, n_pc, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


## RESULT (seed 102926, 80/20 split, 428 test pages)
##   paragraphs only: accuracy 0.786, sens 0.760, spec 0.814, AUC 0.859 (107 PCs)
##   paragraphs + headers:accuracy 0.795, sens 0.791, spec 0.799, AUC 0.866 (193 PCs)
## Headers give a small gain: +0.9 pts accuracy, +0.007 AUC.
## They catch more relevant pages (sens +3.1) but slightly fewer N/A pages (spec -1.5).
## The gain is about 4 test pages, so it could be split noise:
## confirm across several seeds before deciding.




file.size('data/claims-clean-headers-test.RData') / 1e6   # MB
file.size('data/tfidf-headers-test.RData') / 1e6

claims_p  <- select(claims_p,  -text_tmp)
claims_ph <- select(claims_ph, -text_tmp)
save(claims_p, claims_ph, file = 'data/claims-clean-headers-test.RData')

## repeat over several splits ---------------------------------------------

run_seed <- function(s){
  # new train/test pages for this seed, shared by both versions
  set.seed(s)
  sp <- tfidf_p %>%
    select(.id, bclass) %>%
    initial_split(prop = 0.8, strata = bclass)
  tr <- training(sp)$.id
  te <- testing(sp)$.id
  
  bind_rows(
    fit_pcr(tfidf_p,  tr, te) %>% mutate(version = 'paragraphs'),
    fit_pcr(tfidf_ph, tr, te) %>% mutate(version = 'headers')
  ) %>%
    mutate(seed = s)
}

multi <- map_dfr(1:10, run_seed)   # 20 model fits: this may take 10-20 minutes
save(multi, file = 'data/headers-test-seeds.RData')


multi %>%
  filter(.metric %in% c('accuracy', 'roc_auc')) %>%
  select(seed, version, .metric, .estimate) %>%
  pivot_wider(names_from = version, values_from = .estimate) %>%
  mutate(diff = headers - paragraphs) %>%
  group_by(.metric) %>%
  summarise(mean_paragraphs = mean(paragraphs),
            mean_headers    = mean(headers),
            mean_diff       = mean(diff),
            sd_diff         = sd(diff),
            headers_wins    = sum(diff > 0),
            n_seeds         = n())


## RESULT over 10 random 80/20 splits (seeds 1-10)
##   accuracy: paragraphs 0.793 vs headers 0.791 (diff -0.002, sd 0.015, headers win 4/10)
##   ROC AUC:  paragraphs 0.847 vs headers 0.864 (diff +0.018, sd 0.014, headers win 8/10)
## Headers: no change in accuracy at the 0.5 cutoff, consistently better AUC.
## Decision: include headers (h1-h6) in cleaning for the binary model.






