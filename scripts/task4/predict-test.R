## =====================================================================
## PREDICTION SCRIPT (Deliverables 3 and 4)
## Loads the saved binary and multiclass models, predicts every page in
## data/claims-test.RData, and saves pred_df to results/preds-group2.RData.
## No training happens here. Run from the project root in a fresh session.
##
##   Binary model (task 4a):     results/bclass-model.rds
##                               scripts/task4/bclass-functions.R


##   Multiclass model (task 4b): results/mclass-model.rds         
##                               scripts/task4/mclass-functions.R 
## =====================================================================

source('scripts/task4/bclass-functions.R')   # loads packages + predict_bclass()

load('data/claims-test.RData')               # claims_test: .id, text_tmp (929 pages)

# label levels, in the same order as claims_raw (must match exactly)
bclass_levels <- c('N/A: No relevant content.', 'Relevant claim content')
mclass_levels <- c('N/A: No relevant content.', 'Physical Activity',
                   'Possible Fatality', 'Potentially unlawful activity',
                   'Other claim content')


## ---------------------------------------------------------------------
## 1. BINARY PREDICTIONS (task 4a)
## ---------------------------------------------------------------------
bclass_model <- readRDS('results/bclass-model.rds')
bclass_test  <- predict_bclass(bclass_model, claims_test)   # .id, bclass.prob, bclass.pred


## ---------------------------------------------------------------------
## 2. MULTICLASS PREDICTIONS (task 4b, teammate fills this in)
## ---------------------------------------------------------------------
## What the multiclass side needs to provide:
##   - results/mclass-model.rds           the saved model
##   - scripts/task4/mclass-functions.R   defines predict_mclass(model, df)
##       in:  data frame with .id and text_tmp (raw HTML), like claims_test
##       out: tibble with .id and mclass.pred
##            mclass.pred = factor with levels = mclass_levels (above)
##            one row per input page, no NAs
## ---------------------------------------------------------------------
mclass_ready <- file.exists('results/mclass-model.rds') &&
  file.exists('scripts/task4/mclass-functions.R')

if (mclass_ready) {
  source('scripts/task4/mclass-functions.R')
  mclass_model <- readRDS('results/mclass-model.rds')
  mclass_test  <- predict_mclass(mclass_model, claims_test)
} else {
  warning('Multiclass model not found yet: mclass.pred is NA and results will NOT be saved.')
  mclass_test <- tibble(.id = claims_test$.id,
                        mclass.pred = factor(NA, levels = mclass_levels))
}


## ---------------------------------------------------------------------
## 3. COMBINE INTO pred_df
## ---------------------------------------------------------------------
pred_df <- claims_test %>%
  select(.id) %>%
  left_join(select(bclass_test, .id, bclass.pred), by = '.id') %>%
  left_join(select(mclass_test, .id, mclass.pred), by = '.id')


## ---------------------------------------------------------------------
## 4. N/A AGREEMENT (decision rule)
## In the training data, mclass is N/A exactly when bclass is N/A, so any
## test page where the two models disagree is wrong in one column.
## Pick ONE rule below to decide which model to trust.
##
## choosing
##   1. On your held-out pages, collapse multiclass predictions to
##      "N/A vs not N/A" and compute accuracy:
##        na <- 'N/A: No relevant content.'
##        mean((pred_mclass == na) == (truth_mclass == na))
##   2. Compare with the binary model's held-out accuracy: 0.806
##        multiclass higher        -> use rule (b)
##        multiclass lower / tied  -> use rule (a)
##
## Runnig it:
##   - Remove the leading '#' from the code lines of ONE rule only
##   - Keep the '##' comment lines and the other rule commented out
##   - Restart R and rerun this script; all checks should print TRUE
##
## ---------------------------------------------------------------------
na_label <- bclass_levels[1]
disagree <- sum((pred_df$bclass.pred == na_label) != (pred_df$mclass.pred == na_label),
                na.rm = TRUE)
message('Pages where the two models disagree on N/A: ', disagree)

## (a) binary model decides N/A; multiclass picks the claim type
mclass_rel <- predict_mclass_relevant(mclass_model, claims_test)
pred_df <- pred_df %>%
  left_join(mclass_rel, by = ".id") %>%
  mutate(mclass.pred = if_else(bclass.pred == na_label,
                               factor(na_label, levels = mclass_levels),
                               mclass.relevant)) %>%
  select(-mclass.relevant)
  
## (b) multiclass model decides N/A: set bclass.pred from mclass.pred
# pred_df <- pred_df %>%
#   mutate(bclass.pred = factor(if_else(mclass.pred == na_label,
#                                       bclass_levels[1], bclass_levels[2]),
#                               levels = bclass_levels))

## ---------------------------------------------------------------------
## 5. CHECKS (Deliverable 3), then save
## ---------------------------------------------------------------------
checks <- c(
  rows_929        = nrow(pred_df) == 929,
  all_ids         = setequal(pred_df$.id, claims_test$.id),
  no_na_bclass    = !anyNA(pred_df$bclass.pred),
  no_na_mclass    = !anyNA(pred_df$mclass.pred),
  bclass_levels   = identical(levels(pred_df$bclass.pred), bclass_levels),
  mclass_levels   = identical(levels(pred_df$mclass.pred), mclass_levels)
)
print(checks)
print(table(pred_df$bclass.pred, useNA = 'ifany'))
print(table(pred_df$mclass.pred, useNA = 'ifany'))

if (all(checks)) {
  save(pred_df, file = 'results/preds-group2.RData')
  message('Saved results/preds-group2.RData')
} else {
  message('Not saved: fix the FALSE checks above first.')
}