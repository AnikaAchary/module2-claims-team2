## MULTICLASS MODEL FUNCTIONS (task 4b)
## Needs bclass-functions.R sourced first (cleaning + TF-IDF functions).

# raw pages (.id, text_tmp) -> .id, mclass.pred
predict_mclass <- function(model, .df) {
  cl <- clean_pages(.df, model$css)
  tk <- tokenize_pages(cl)
  x  <- make_dtm(tk, cl$.id, model$vocab)

  lab <- as.character(predict(model$fit, s = model$lambda, newx = x, type = "class"))
  lab[!cl$parsed] <- model$fallback   # non-HTML pages -> N/A

  tibble(.id = cl$.id,
         mclass.pred = factor(lab, levels = model$levels))
}

# raw pages (.id, text_tmp) -> .id, mclass.relevant
# most likely class other than N/A, used by the N/A rule in predict-test.R
predict_mclass_relevant <- function(model, .df) {
  cl <- clean_pages(.df, model$css)
  tk <- tokenize_pages(cl)
  x  <- make_dtm(tk, cl$.id, model$vocab)

  p   <- predict(model$fit, s = model$lambda, newx = x, type = "response")[, , 1]
  p   <- p[, colnames(p) != model$fallback]
  rel <- colnames(p)[max.col(p, ties.method = "first")]

  tibble(.id = cl$.id,
         mclass.relevant = factor(rel, levels = model$levels))
}
