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


