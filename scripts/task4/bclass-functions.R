## =====================================================================
## BINARY MODEL FUNCTIONS (task 4a)
## Everything needed to turn raw HTML into bclass predictions with the
## saved model results/bclass-model.rds.
## Sourced by: scripts/task4/binary-model-development.R
##             scripts/task4/predict-test.R
## =====================================================================

library(tidyverse)
library(tidytext)    # unnest_tokens(), stop_words
library(textstem)    # lemmatize_words()
library(rvest)       # read_html(), html_elements(), html_text2()
library(qdapRegex)   # rm_url(), rm_email()
library(Matrix)      # sparseMatrix()
library(glmnet)      # predict() for the lasso fit

# HTML tags read from each page: paragraphs + headers (decided in Phase 1)
css_headers <- 'p, h1, h2, h3, h4, h5, h6'


## ---------------------------------------------------------------------
## CLEANING
## ---------------------------------------------------------------------

# one page of HTML -> one clean lowercase string
# same steps as parse_fn() in scripts/preprocessing.R, but .css picks the tags
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

# clean EVERY page in a data frame with a text_tmp column
# - pages containing '<!' or '<html' are parsed as HTML
# - anything else (JSON, PDFs, scraper errors) gets empty text and parsed = FALSE
# - no page is ever dropped
clean_pages <- function(.df, .css = css_headers){
  .df %>%
    mutate(parsed = str_detect(text_tmp, regex('<!|<html', ignore_case = TRUE))) %>%
    rowwise() %>%
    mutate(text_clean = if (parsed) {
      tryCatch(parse_fn2(text_tmp, .css), error = function(e) '')
    } else '') %>%
    ungroup() %>%
    mutate(text_clean = coalesce(text_clean, '')) %>%
    select(-text_tmp)
}


## ---------------------------------------------------------------------
## FEATURES
## ---------------------------------------------------------------------

# clean text -> lemma counts per page (.id, token.lem, n)
tokenize_pages <- function(.clean){
  .clean %>%
    select(.id, text_clean) %>%
    unnest_tokens(output = token, input = text_clean, token = 'words',
                  stopwords = str_remove_all(stop_words$word, '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, token.lem, name = 'n')
}

# vocabulary + IDF weights, learned from TRAINING pages only
build_vocab <- function(tokens, n_pages, min_docs = 5){
  tokens %>%
    count(token.lem, name = 'docs') %>%
    filter(docs >= min_docs) %>%
    mutate(idf = log(n_pages / docs)) %>%
    arrange(token.lem)
}

# TF-IDF matrix for any pages, using a saved vocabulary
# rows follow `ids` exactly (blank pages = zero rows); columns follow the vocabulary
make_dtm <- function(tokens, ids, vocab){
  tf <- tokens %>%
    filter(token.lem %in% vocab$token.lem) %>%
    group_by(.id) %>%
    mutate(tf = n / sum(n)) %>%
    ungroup() %>%
    left_join(select(vocab, token.lem, idf), by = 'token.lem') %>%
    mutate(i = match(.id, ids), j = match(token.lem, vocab$token.lem)) %>%
    filter(!is.na(i))
  
  sparseMatrix(i = tf$i, j = tf$j, x = tf$tf * tf$idf,
               dims = c(length(ids), nrow(vocab)),
               dimnames = list(ids, vocab$token.lem))
}


## ---------------------------------------------------------------------
## PREDICTION
## ---------------------------------------------------------------------

# raw pages (.id, text_tmp) -> .id, bclass.prob, bclass.pred
# uses only what is stored in the saved model:
#   css, vocab, fit, lambda, threshold, levels, fallback
predict_bclass <- function(model, .df){
  cl <- clean_pages(.df, model$css)
  tk <- tokenize_pages(cl)
  x  <- make_dtm(tk, cl$.id, model$vocab)
  
  p   <- as.numeric(predict(model$fit, s = model$lambda, newx = x, type = 'response'))
  lab <- if_else(p > model$threshold, model$levels[2], model$levels[1])
  lab[!cl$parsed] <- model$fallback   # non-HTML pages -> N/A
  
  tibble(.id = cl$.id,
         bclass.prob = p,
         bclass.pred = factor(lab, levels = model$levels))
}