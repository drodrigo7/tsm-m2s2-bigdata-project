library(haven)
library(dplyr)
library(tidyr)
library(purrr)
library(randomForest)
library(caret)

# Suppress warnings
options(warn = -1)

# Raw Data
# ==================================================
path <- './data/raw/enaho/'

df_raw <- list()
for (y in 2004:2024) {
  df_chunk <- read_dta(
    file = paste0(path, 'sumaria-', y, '.dta'),
    col_select = c(
      'conglome',
      'vivienda',
      'hogar',
      'ubigeo',
      'dominio',
      'estrato',
      'percepho',
      'totmieho',
      'mieperho',
      'ga04hd',
      'ia01hd',
      'ingmo1hd',
      'ingmo2hd',
      'inghog1d',
      'inghog2d',
      'gashog1d',
      'gashog2d',
      'pobreza'
    )
  )
  df_chunk$year <- y
  df_raw[[as.character(y)]] <- df_chunk
}

df <- bind_rows(df_raw)
print(dim(df))
print(head(df, 10))

# Feature Engineering
# ==================================================
# Log transformation for monetary variables
monetary_vars <- c(
  'ia01hd',
  'ga04hd',
  'ingmo1hd',
  'ingmo2hd',
  'inghog1d',
  'inghog2d',
  'gashog1d',
  'gashog2d'
)

for (c in monetary_vars) {
  if (c %in% colnames(df)) {
    df[[paste0('log_', c)]] <- log1p(df[[c]])
  }
}

# Relative size (dependency ratio)
df <- df %>%
  mutate(dependencia = totmieho / pmax(percepho, 1))

# Urban or rural area
df <- df %>%
  mutate(
    area = ifelse(estrato <= 4, 'urban', 'rural'),
    area = factor(area)
  )

# Domain categories
dominio_labels <- c(
  '1' = 'north_coast',
  '2' = 'central_coast',
  '3' = 'south_coast',
  '4' = 'north_highlands',
  '5' = 'central_highlands',
  '6' = 'south_highlands',
  '7' = 'jungle',
  '8' = 'metropolitan_lima'
)

df <- df %>%
  mutate(dominio_cat = factor(dominio_labels[as.character(dominio)]))

# Poverty categories
pobreza_labels <- c(
  '1' = 'poverty',
  '2' = 'poverty',
  '3' = 'no poverty'
)

df <- df %>%
  mutate(pobreza_cat = factor(pobreza_labels[as.character(pobreza)]))

# Renaming columns
df <- df %>%
  rename(
    # Household identifiers
    cluster_id = conglome,
    dwelling_id = vivienda,
    household_id = hogar,
    geo_code = ubigeo,
    # Geography
    domain_code = dominio,
    domain = dominio_cat,
    stratum = estrato,
    area_type = area,
    # Household composition
    income_receivers = percepho,
    household_total_size = totmieho,
    household_members = mieperho,
    dependency_ratio = dependencia,
    # Log-transformed
    log_imputed_rent_income = log_ia01hd,
    log_imputed_rent_expense = log_ga04hd,
    log_monetary_income_gross = log_ingmo1hd,
    log_monetary_income_net = log_ingmo2hd,
    log_household_income_gross = log_inghog1d,
    log_household_income_net = log_inghog2d,
    log_household_expense_monetary = log_gashog1d,
    log_household_expense_total = log_gashog2d,
    # Poverty
    poverty_code = pobreza,
    poverty_status = pobreza_cat,
    # Time
    year = year
  )

# Target balance
table(df$poverty_status, useNA = "ifany")
prop.table(table(df$poverty_status, useNA = "ifany"))

# Modeling
# ==================================================
df_model <- df %>%
  select(-cluster_id, -dwelling_id, -household_id, -geo_code)

df_model <- df_model %>%
  mutate(poverty_binary = as.integer(poverty_status == 'poverty'))

model_columns <- c(
  'income_receivers',
  'household_total_size',
  'household_members',
  'dependency_ratio',
  'log_imputed_rent_income',
  'log_imputed_rent_expense',
  'log_household_income_gross',
  'log_household_expense_total',
  'area_type',
  'domain',
  'poverty_binary'
)

df_model <- df_model %>%
  select(all_of(model_columns))

# Prepare data for modeling
X <- df_model %>% select(-poverty_binary)
y <- df_model$poverty_binary

# Create dummy variables (one-hot encoding)
# Note: model.matrix automatically drops first level
dummy_formula <- as.formula(paste("~", paste(names(X), collapse = " + ")))
X_encoded <- model.matrix(dummy_formula, data = X)[, -1]  # Remove intercept
X_encoded <- as.data.frame(X_encoded)

# Train-test split
set.seed(42)
train_index <- createDataPartition(y, p = 0.70, list = FALSE)

X_train <- X_encoded[train_index, ]
X_test <- X_encoded[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Random Forest Classifier
# Note: In R's randomForest, class weights are handled differently
# We'll use sampsize to balance classes
n_poverty <- sum(y_train == 1)
n_no_poverty <- sum(y_train == 0)

# Use the minimum class size for balanced sampling
min_class_size <- min(n_poverty, n_no_poverty)

set.seed(42)
rf <- randomForest(
  x = X_train,
  y = factor(y_train, levels = c(0, 1), labels = c("no poverty", "poverty")),
  ntree = 100,
  sampsize = c(min_class_size, min_class_size),  # Balance classes
  importance = TRUE
)

# Predictions
y_pred <- predict(rf, X_test)
y_prob <- predict(rf, X_test, type = "prob")[, "poverty"]

# Convert predictions back to numeric for some metrics
y_pred_numeric <- as.integer(y_pred == "poverty")

# Evaluation metrics
accuracy <- mean(y_pred_numeric == y_test)
print(paste("Accuracy:", accuracy))

# Classification report (manual)
confusion <- table(Predicted = y_pred, Actual = factor(y_test, levels = c(0, 1), 
                                                        labels = c("no poverty", "poverty")))
print(confusion)

# Precision, Recall, F1 for each class
precision_poverty <- confusion["poverty", "poverty"] / sum(confusion["poverty", ])
recall_poverty <- confusion["poverty", "poverty"] / sum(confusion[, "poverty"])
f1_poverty <- 2 * (precision_poverty * recall_poverty) / (precision_poverty + recall_poverty)

precision_no_poverty <- confusion["no poverty", "no poverty"] / sum(confusion["no poverty", ])
recall_no_poverty <- confusion["no poverty", "no poverty"] / sum(confusion[, "no poverty"])
f1_no_poverty <- 2 * (precision_no_poverty * recall_no_poverty) / (precision_no_poverty + recall_no_poverty)

cat("\nClassification Report:\n")
cat("no poverty - Precision:", round(precision_no_poverty, 2), 
    "Recall:", round(recall_no_poverty, 2), 
    "F1:", round(f1_no_poverty, 2), "\n")
cat("poverty - Precision:", round(precision_poverty, 2), 
    "Recall:", round(recall_poverty, 2), 
    "F1:", round(f1_poverty, 2), "\n")

# ROC-AUC
library(pROC)
roc_obj <- roc(y_test, y_prob)
auc_score <- auc(roc_obj)
print(paste("ROC-AUC:", round(auc_score, 4)))

# Confusion Matrix Plot
library(ggplot2)
confusion_df <- as.data.frame(confusion)

ggplot(confusion_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()

# Feature Importance
feature_importance <- data.frame(
  feature = rownames(importance(rf)),
  importance = importance(rf)[, "MeanDecreaseGini"]
) %>%
  arrange(desc(importance))

print(head(feature_importance, 10))

# Feature Importance Plot
ggplot(head(feature_importance, 10), aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Feature Importances", x = "Feature", y = "Importance") +
  theme_minimal()