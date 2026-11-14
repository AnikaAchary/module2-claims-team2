require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

parse_fn <- function(.html, tags = 'p'){
  read_html(.html) %>%
    html_elements(tags) %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

load('data/claims-raw.RData')

pages <- claims_raw %>% filter(str_detect(text_tmp, '<!')) %>% ungroup()
set.seed(197)
split_indices <- split(seq_len(nrow(pages)), pages$bclass)
train <- unlist(lapply(split_indices, function(i) sample(i, floor(.6 * length(i)))))
remaining <- lapply(split_indices, function(i) setdiff(i, train))
validation <- unlist(lapply(remaining, function(i) sample(i, floor(length(i) / 2))))
test <- setdiff(seq_len(nrow(pages)), c(train, validation))
y <- as.integer(pages$bclass == 'Relevant claim content')
stopifnot(length(unique(y)) == 2)

texts <- list(
  'Paragraphs only' = map_chr(pages$text_tmp, parse_fn),
  'Paragraphs + headings' = map_chr(pages$text_tmp, parse_fn,
                                  tags = 'p, h1, h2, h3, h4, h5, h6')
)

grid <- c(5, 10, 20, 30, 50)
metrics <- list()
for (strategy in names(texts)) {
  message('Fitting: ', strategy)
  text <- texts[[strategy]]
  tokens <- tibble(doc = seq_along(text), text_clean = text) %>%
    unnest_tokens(token, text_clean, token = 'words',
                  stopwords = str_remove_all(stop_words$word, '[[:punct:]]'))
  words <- unique(tokens$token)
  lemmas <- setNames(lemmatize_words(words), words)
  counts <- tokens %>% mutate(term = unname(lemmas[token])) %>%
    filter(str_length(term) > 2) %>% count(doc, term, name = 'n')
  vocabulary <- counts %>% filter(doc %in% train) %>%
    count(term, name = 'df') %>%
    filter(df >= 3, df < length(train)) %>%
    arrange(desc(df), term) %>% slice_head(n = 1000) %>%
    mutate(idf = log(length(train) / df))
  weighted <- counts %>% group_by(doc) %>% mutate(tf = n / sum(n)) %>%
    ungroup() %>% inner_join(vocabulary, by = 'term')
  x <- matrix(0, length(text), nrow(vocabulary))
  x[cbind(weighted$doc, match(weighted$term, vocabulary$term))] <-
    weighted$tf * weighted$idf
  pca <- prcomp(x[train, , drop = FALSE], center = TRUE,
                scale. = FALSE, rank. = max(grid))
  scores <- as.data.frame(predict(pca, x))
  fits <- lapply(grid, function(k) {
    glm(y ~ ., data = cbind(y = y[train], scores[train, seq_len(k), drop = FALSE]),
        family = binomial())
  })
  validation_accuracy <- vapply(seq_along(grid), function(j) {
    probability <- predict(fits[[j]], scores[validation, , drop = FALSE], type = 'response')
    mean((probability >= .5) == y[validation])
  }, numeric(1))
  best <- which.max(validation_accuracy)
  probability <- predict(fits[[best]], scores[test, , drop = FALSE], type = 'response')
  correct <- sum((probability >= .5) == y[test])
  metrics[[strategy]] <- tibble(strategy, components = grid[best],
                                accuracy = correct / length(test))
}
metrics <- bind_rows(metrics)

accuracy_table <- metrics %>%
  gt::gt() %>%
  gt::tab_header(title = 'Binary prediction accuracy: logistic PCR') %>%
  gt::cols_label(strategy = 'HTML content', components = 'Principal components',
                 accuracy = 'Test accuracy') %>%
  gt::fmt_percent(accuracy, decimals = 2)
print(accuracy_table)
