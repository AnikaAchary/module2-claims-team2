# TASK 3: neural network for binary classification
# By: Anika Achary
# goal: match/beat Task 1's logistic PCR accuracy on same test split

reticulate::py_require("tensorflow")
library(tidyverse)
library(tidymodels)
library(keras3)
library(tensorflow)

## DATA

# swap with header version once task 1 is pushed to compare against PCR
load('data/claims-clean-example.RData')

# same partition as starter code
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

# shuffle training rows so keras's validation_split doesn't end up 
# with a weird chunk of the data (takes last 20% of rows)
train_df <- training(partitions) %>% slice_sample(prop = 1)
test_df <- testing(partitions)

train_text <- train_df %>% pull(text_clean)
train_labels <- train_df %>% pull(bclass) %>% as.numeric() - 1

test_text <- test_df %>% pull(text_clean)
test_labels <- test_df %>% pull(bclass)

train_x <- tf$constant(matrix(train_text, ncol = 1))
test_x  <- tf$constant(matrix(test_text, ncol = 1))

class_labels <- levels(test_labels)

## PREPROCESSING LAYER (text -> tf-idf vector)

# makes weight initialization repeatable so results don't jumpy around
# run to run
tensorflow::set_random_seed(110122)

preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = 2,       # 2 for bigrams  
  max_tokens = 20000,     
  output_mode = 'tf_idf'
)

preprocess_layer %>% adapt(train_x)

## MODEL

inputs <- layer_input(shape = 1, dtype = 'string')

outputs <- inputs %>%
  preprocess_layer() %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model <- keras_model(inputs, outputs)

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = 'binary_accuracy'
)

## TRAIN

# train for up to 30 epochs but stop once validation loss stops improving,
# then roll back to the best epoch

history <- model %>%
  fit(train_x,
      train_labels,
      validation_split = 0.2,
      epochs = 30,
      callbacks = list(
        callback_early_stopping(monitor = 'val_loss',
                                patience = 3,
                                restore_best_weights = TRUE)
      ))

plot(history)   # look for where val loss bottoms out / starts rising

## EVALUATE ON HELD-OUT TEST SET

pred_probs <- predict(model, test_x) %>% as.numeric()

pred_classes <- factor(pred_probs > 0.5,
                       levels = c(FALSE, TRUE),
                       labels = class_labels)

results <- tibble(truth = test_labels,
                  estimate = pred_classes,
                  prob = pred_probs)

class_metrics <- metric_set(accuracy, sensitivity, specificity)

results %>% class_metrics(truth = truth, estimate = estimate)
results %>% roc_auc(truth = truth, prob, event_level = 'second')

## SAVE

save_model(model, 'results/task3-nn.keras')

# later, load with:
# load_model('results/task3-nn.keras')