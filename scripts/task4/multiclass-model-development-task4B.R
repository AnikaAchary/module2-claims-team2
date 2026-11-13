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

## MODEL COMPARISON ----------------------------------------------------------
panel <- metric_set(sensitivity, specificity, accuracy)
na_label <- mlv[1]

# macro metrics + N/A vs not accuracy
score_m <- function(truth, pred) {
  df <- tibble(mclass = truth, mclass.pred = pred)
  df %>%
    panel(truth = mclass, estimate = mclass.pred, estimator = "macro") %>%
    bind_rows(tibble(.metric = "na_accuracy", .estimator = "binary",
                     .estimate = mean((pred == na_label) == (truth == na_label))))
}

# one split, vocab from training pages only
fit_m <- function(train_ids, test_ids, method = "lasso", alpha = 1) {
  tr <- train_clean %>% filter(.id %in% train_ids, parsed) %>% arrange(.id)
  te <- train_clean %>% filter(.id %in% test_ids) %>% arrange(.id)

  tok_tr <- tok %>% filter(.id %in% tr$.id)
  tok_te <- tok %>% filter(.id %in% te$.id)

  v    <- build_vocab(tok_tr, n_pages = nrow(tr))
  x_tr <- make_dtm(tok_tr, tr$.id, v)
  x_te <- make_dtm(tok_te, te$.id, v)

  if (method == "pca") {
    proj <- projection_fn(.dtm = x_tr, .prop = 0.7)
    x_tr <- as.matrix(proj$data)
    x_te <- as.matrix(reproject_fn(.dtm = x_te, proj))
  }

  set.seed(67)
  cv <- cv.glmnet(x = x_tr, y = tr$mclass, family = "multinomial", alpha = alpha)
  pred <- as.character(predict(cv, s = "lambda.min", newx = x_te, type = "class"))
  pred[!te$parsed] <- na_label   # non-HTML pages -> N/A

  score_m(te$mclass, factor(pred, levels = mlv))
}

# same 10 splits for every model, run in parallel
evaluate_m <- function(seeds = 67:76, ...) {
  out <- parallel::mclapply(seeds, function(s) {
    set.seed(s)
    sp <- train_clean %>%
      arrange(.id) %>%
      select(.id, mclass) %>%
      initial_split(prop = 0.8, strata = mclass)
    fit_m(training(sp)$.id, testing(sp)$.id, ...) %>%
      mutate(seed = s)
  }, mc.cores = 5)
  bind_rows(out)
}


res_pca   <- evaluate_m(method = "pca", alpha = 0.3) %>% mutate(model = "PCA + multinomial (baseline)")
res_lasso <- evaluate_m(method = "lasso", alpha = 1) %>% mutate(model = "multinomial lasso")
res_enet  <- evaluate_m(method = "lasso", alpha = 0.5) %>% mutate(model = "multinomial elastic net")

results_m <- bind_rows(res_pca, res_lasso, res_enet)
save(results_m, file = "results/multiclass-comparison.RData")

results_m %>%
  group_by(model, .metric) %>%
  summarise(mean = mean(.estimate), sd = sd(.estimate), .groups = "drop") %>%
  pivot_wider(names_from = .metric, values_from = c(mean, sd)) %>%
  print(width = Inf)

## RESULT (10 splits): lasso and elastic net beat the PCA baseline
##   baseline 0.771 (macro sens 0.617) | lasso 0.779 (0.662) | elastic net 0.781 (0.662)
##   lasso vs elastic net is within split noise (sd ~0.012)
## DECISION: multinomial lasso (same method as the binary model)
## N/A vs not accuracy 0.786 < binary 0.806 -> use rule (a) in predict-test.R

## REFIT + SAVE --------------------------------------------------------------
train_fit   <- train_clean %>% filter(parsed) %>% arrange(.id)   # 2,146 HTML pages
tok_fit     <- tok %>% filter(.id %in% train_fit$.id)
vocab_final <- build_vocab(tok_fit, n_pages = nrow(train_fit))
x_all       <- make_dtm(tok_fit, train_fit$.id, vocab_final)
dim(x_all)

set.seed(67)
cv_final <- cv.glmnet(x = x_all, y = train_fit$mclass, family = "multinomial", alpha = 1)

mclass_model <- list(
  css      = css_headers,
  vocab    = vocab_final,
  fit      = cv_final,
  lambda   = cv_final$lambda.min,
  levels   = mlv,
  fallback = mlv[1]   # non-HTML pages -> N/A
)

saveRDS(mclass_model, file = "results/mclass-model.rds")
file.size("results/mclass-model.rds") / 1e6   # MB
