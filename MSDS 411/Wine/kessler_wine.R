# Unit 3 Project
# MSDS 411 Section 56
# Alan Kessler

library(RColorBrewer) # nolint
library(car)
library(reshape2)
library(ggplot2)
library(MASS)
library(pscl)
library(dplyr)
library(caret)
library(rpart)

# Setting colors for plots
pal <- brewer.pal(3, "Set1")

# Load data
training <- read.csv("Wine_Training.csv")
testing <- read.csv("Wine_Test.csv")

# Section 1: Data Exploration

# Analyze target distribution
ggplot(training, aes(TARGET)) +
  geom_bar(fill = pal[2], color = "black") +
  scale_x_continuous(breaks = seq(0, 8, by = 1)) +
  ggtitle("Training Target Distribution") +
  ylab("Number of Wines") +
  xlab("Cases of Wines Sold")

# Compare variance to mean for target values greater than zero
target_analysis <- training %>%
  filter(TARGET > 0) %>%
  mutate(TARGET_M1 = TARGET - 1) %>%
  summarise(M = mean(TARGET_M1), V = var(TARGET_M1))

print(target_analysis)

# Visualize missing values in the data
MissingCount <- training %>%
  mutate_all(is.na) %>%
  summarise_all(sum) %>%
  melt(id.vars = NULL) %>%
  mutate(value = value / nrow(training)) %>%
  filter(value > 0)

ggplot(MissingCount, aes(x = variable, weight = value)) +
  geom_bar(color = "black", fill = pal[1]) +
  xlab("Variable (Those with Missing Values)") +
  ylab("Missing Fraction") +
  ggtitle("Missing Values - Training") +
  theme(axis.text.x = element_text(angle = 10, hjust = 1, size = 8))

# Training predictor correlation
corr <- cor(training[, !names(training) %in% c("INDEX")], 
            use = "pairwise.complete.obs")
corr_melt <- melt(round(corr, 2))

ggplot(data = corr_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 2.75) +
  scale_fill_gradient2(low = pal[2], high = pal[1], mid = "white", midpoint = 0,
                       limit = c(-1, 1), space = "Lab", name = "Correlation") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.title = element_text(size = 8),
        legend.key.size = unit(0.75, "line")) +
  xlab("") +
  ylab("") +
  ggtitle("Correlation - Training Data")

# Merge training and testing for analyzing predictors
df1 <- training %>%
  select(-TARGET, -INDEX) %>%
  mutate(Data = "Training")

df2 <- testing %>%
  select(-TARGET, -INDEX) %>%
  mutate(Data = "Testing")

df <- rbind(df1, df2) %>%
  melt(id.vars = "Data") %>%
  na.omit()

# Box plots of predictors
ggplot(df, aes(x = variable, y = value, color = Data)) +
  xlab("Variable") +
  ylab("Value") +
  ggtitle("Box Plots for Continuous Predictor Variables") +
  geom_boxplot() +
  facet_wrap(~variable, scales = "free") +
  theme(strip.text.x = element_blank())

# Histograms of predictors
ggplot(df, aes(value, fill = Data)) +
  xlab("Variable Value") +
  ylab("Count") +
  ggtitle("Histograms for Predictor Variables") +
  geom_histogram(bins = 30, color = "black") +
  facet_wrap(~variable, scales = "free") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Univariate relationships with the target
# Different approach having y on x-axis
train_melt <- training %>%
  select(-INDEX) %>%
  melt(., id.vars = "TARGET") %>%
  na.omit() %>%
  group_by(variable, TARGET) %>%
  summarise(value = mean(value))

ggplot(train_melt, aes(x = as.factor(TARGET), y = value, group = 1)) +
  geom_line() +
  facet_wrap(~variable, scales = "free") +
  ggtitle("Univariate Relationship with Continuous Target") +
  xlab("Target") +
  ylab("Mean Predictor Variable Value") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Part 2: Data Preparation

# Function to impute a variable with the mean
impute_mean <- function(v, base) {
  ifelse(is.na(v), mean(base, na.rm = TRUE), v)
}

# Function to create missing indicator
miss_ind <- function(v) {
  ifelse(is.na(v), 1, 0)
}

# Function to cap/flr a variable
cap_flr <- function(v, base, mi = 0.025, ma = 0.975) {
  ifelse(v > quantile(base, na.rm = TRUE, ma), quantile(base, na.rm = TRUE, ma),
         ifelse(v < quantile(base, na.rm = TRUE, mi), 
                quantile(base, na.rm = TRUE, mi), v))
}

# Function to create an indicator if capping or flooring is applied
cap_flr_ind <- function(v, base, mi = 0.025, ma = 0.975) {
  ifelse(v > quantile(base, na.rm = TRUE, ma), 1,
         ifelse(v < quantile(base, na.rm = TRUE, mi), 1, 0))
}

# Convert stars to a factor where missing is its own category
# Create a version of LabelAppeal as a factor to compete with the numeric type
# Impute missing with the training mean
# Create missing indicator variables
# Cap and floor most variables at 2.5% and 97.5% of training data
# Create indicators of whether capping or flooring occurred
train_process1 <- training %>%
  select(-INDEX) %>%
  mutate(STARS_NR = ifelse(is.na(STARS), 1, 0),
         STARS_1 = ifelse(STARS == 1 & STARS_NR == 0, 1, 0),
         STARS_3 = ifelse(STARS == 3 & STARS_NR == 0, 1, 0),
         STARS_4 = ifelse(STARS == 4 & STARS_NR == 0, 1, 0),
         LabelAppeal_N2 = ifelse(LabelAppeal == -2, 1, 0),
         LabelAppeal_N1 = ifelse(LabelAppeal == -1, 1, 0),
         LabelAppeal_1 = ifelse(LabelAppeal == 1, 1, 0),
         LabelAppeal_2 = ifelse(LabelAppeal == 2, 1, 0),
         ResidualSugar_imp = impute_mean(ResidualSugar, training$ResidualSugar),
         Chlorides_imp = impute_mean(Chlorides, training$Chlorides),
         FreeSulfurDioxide_imp = impute_mean(FreeSulfurDioxide,
                                             training$FreeSulfurDioxide),
         TotalSulfurDioxide_imp = impute_mean(TotalSulfurDioxide,
                                              training$TotalSulfurDioxide),
         pH_imp = impute_mean(pH, training$pH),
         Sulphates_imp = impute_mean(Sulphates, training$Sulphates),
         Alcohol_imp = impute_mean(Alcohol, training$Alcohol),
         ResidualSugar_mind = miss_ind(ResidualSugar),
         Chlorides_mind = miss_ind(Chlorides),
         FreeSulfurDioxide_mind = miss_ind(FreeSulfurDioxide),
         TotalSulfurDioxide_mind = miss_ind(TotalSulfurDioxide),
         pH_mind = miss_ind(pH),
         Sulphates_mind = miss_ind(Sulphates),
         Alcohol_mind = miss_ind(Alcohol),
         FixedAcidity_cf = cap_flr(FixedAcidity, training$FixedAcidity),
         FixedAcidity_cfind = cap_flr_ind(FixedAcidity, training$FixedAcidity),
         VolatileAcidity_cf = cap_flr(VolatileAcidity,
                                      training$VolatileAcidity),
         VolatileAcidity_cfind = cap_flr_ind(VolatileAcidity, 
                                             training$VolatileAcidity),
         CitricAcid_cf = cap_flr(CitricAcid, training$CitricAcid),
         CitricAcid_cfind = cap_flr_ind(CitricAcid, training$CitricAcid),
         ResidualSugar_cf = cap_flr(ResidualSugar_imp, training$ResidualSugar),
         ResidualSugar_cfind = cap_flr_ind(ResidualSugar_imp, 
                                           training$ResidualSugar),
         Chlorides_cf = cap_flr(Chlorides_imp, training$Chlorides),
         Chlorides_cfind = cap_flr_ind(Chlorides_imp, training$Chlorides),
         FreeSulfurDioxide_cf = cap_flr(FreeSulfurDioxide_imp,
                                        training$FreeSulfurDioxide),
         FreeSulfurDioxide_cfind = cap_flr_ind(FreeSulfurDioxide_imp,
                                               training$FreeSulfurDioxide),
         TotalSulfurDioxide_cf = cap_flr(TotalSulfurDioxide_imp,
                                         training$TotalSulfurDioxide),
         TotalSulfurDioxide_cfind = cap_flr_ind(TotalSulfurDioxide_imp,
                                                training$TotalSulfurDioxide),
         Density_cf = cap_flr(Density, training$Density),
         Density_cfind = cap_flr_ind(Density, training$Density),
         pH_cf = cap_flr(pH_imp, training$pH),
         pH_cfind = cap_flr_ind(pH_imp, training$pH),
         Sulphates_cf = cap_flr(Sulphates_imp, training$Sulphates),
         Sulphates_cfind = cap_flr_ind(Sulphates_imp, training$Sulphates),
         Alcohol_cf = cap_flr(Alcohol_imp, training$Alcohol),
         Alcohol_cfind = cap_flr_ind(Alcohol_imp, training$Alcohol)) %>%
  select(-STARS, -ResidualSugar, -Chlorides, -FreeSulfurDioxide, -pH, -Alcohol,
         -Sulphates, -TotalSulfurDioxide, -FixedAcidity, -VolatileAcidity,
         -CitricAcid, -ResidualSugar_imp, -Chlorides_imp, -Density, -pH_imp,
         -FreeSulfurDioxide_imp, -TotalSulfurDioxide_imp, -Sulphates_imp,
         -Alcohol_imp, -LabelAppeal)

vars_categorical1 <- c("ResidualSugar_mind", "Chlorides_mind",
                       "FreeSulfurDioxide_mind", "pH_mind", "Alcohol_mind",
                       "Sulphates_mind", "TotalSulfurDioxide_mind")

vars_categorical2 <- c("ResidualSugar_cfind", "Chlorides_cfind",
                       "FreeSulfurDioxide_cfind", "pH_cfind", "Alcohol_cfind",
                       "Sulphates_cfind", "TotalSulfurDioxide_cfind", 
                       "FixedAcidity_cfind", "VolatileAcidity_cfind", 
                       "CitricAcid_cfind")

vars_continuous <- c("ResidualSugar_cf", "Chlorides_cf", "FreeSulfurDioxide_cf",
                     "pH_cf", "Alcohol_cf", "Sulphates_cf",
                     "TotalSulfurDioxide_cf", "FixedAcidity_cf",
                     "VolatileAcidity_cf", "CitricAcid_cf")

# Univariate relationship with target: Categorical group 1
train_melt <- train_process1 %>%
  select(vars_categorical1, TARGET) %>%
  melt(., id.vars = "TARGET") %>%
  group_by(variable, value) %>%
  summarise(n = n(),
            m = mean(TARGET),
            v = var(TARGET)) %>%
  ungroup() %>%
  mutate(m_plus = m + qnorm(0.95) * sqrt(v / n),
         m_minus = m - qnorm(0.95) * sqrt(v / n))

ggplot(data = train_melt, aes(value, m)) +
  geom_line() +
  geom_line(y = train_melt$m_plus, color = pal[1]) +
  geom_line(y = train_melt$m_minus, color = pal[1]) +
  facet_wrap(~variable, scales = "free") +
  scale_x_continuous(breaks = seq(0, 13, by = 1)) +
  xlab("Variable Value") +
  ylab("Average Cases Sold") +
  ggtitle("Training Univariate Cases Sold with 90% CI") +
  theme(strip.text.x = element_text(size = 6),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Univariate relationship with target: Categorical group 2
train_melt <- train_process1 %>%
  select(vars_categorical2, TARGET) %>%
  melt(., id.vars = "TARGET") %>%
  group_by(variable, value) %>%
  summarise(n = n(),
            m = mean(TARGET),
            v = var(TARGET)) %>%
  ungroup() %>%
  mutate(m_plus = m + qnorm(0.95) * sqrt(v / n),
         m_minus = m - qnorm(0.95) * sqrt(v / n))

ggplot(data = train_melt, aes(value, m)) +
  geom_line() +
  geom_line(y = train_melt$m_plus, color = pal[1]) +
  geom_line(y = train_melt$m_minus, color = pal[1]) +
  facet_wrap(~variable, scales = "free") +
  scale_x_continuous(breaks = seq(0, 13, by = 1)) +
  xlab("Variable Value") +
  ylab("Average Cases Sold") +
  ggtitle("Training Univariate Cases Sold with 90% CI") +
  theme(strip.text.x = element_text(size = 6),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Univariate relationships with the target
# Different approach having y on x-axis
train_melt <- train_process1 %>%
  select(TARGET, vars_continuous) %>%
  melt(., id.vars = "TARGET") %>%
  na.omit() %>%
  group_by(variable, TARGET) %>%
  summarise(value = mean(value))

ggplot(train_melt, aes(x = as.factor(TARGET), y = value, group = 1)) +
  geom_line() +
  facet_wrap(~variable, scales = "free") +
  ggtitle("Univariate Relationship with Continuous Target") +
  xlab("Target") +
  ylab("Mean Predictor Variable Value") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Part 3 Build Models

# Split training data to observe how models fit on holdout data 
set.seed(6174)
trainIndex <- createDataPartition(train_process1$TARGET, p = 0.7, list = FALSE)

train_prelim <- train_process1[ trainIndex, ]
train_validation <- train_process1[-trainIndex, ]

# Validation mse function
val_mse <- function(model, log = FALSE) {
  if (log == TRUE) {
    predictions <- exp(predict(model, train_validation, type = "response"))
  } else {
    predictions <- predict(model, train_validation, type = "response")
  }
  mse <- data.frame(predictions, train_validation$TARGET)
  names(mse) <- c("predictions", "TARGET")
  mse <- mse %>%
    mutate(sq_error = (TARGET - predictions) ^ 2) %>%
    summarise(mse_model1 = mean(sq_error))
  return(mse[1, 1])
}

# Training mse function
train_mse <- function(model, log = FALSE) {
  if (log == TRUE) {
    predictions <- exp(predict(model, train_prelim, type = "response"))
  } else {
    predictions <- predict(model, train_prelim, type = "response")
  }
  mse <- data.frame(predictions, train_prelim$TARGET)
  names(mse) <- c("predictions", "TARGET")
  mse <- mse %>%
    mutate(sq_error = (TARGET - predictions) ^ 2) %>%
    summarise(mse_model1 = mean(sq_error))
  return(mse[1, 1])
}

# Function for general result printing
model_diag <- function(model) {
  print(summary(model))
  print("VIFs:")
  print(vif(model))
  print(paste0("AIC: ", AIC(model)))
  print(paste0("Training MSE: ", train_mse(model)))
  print(paste0("Validation MSE: ", val_mse(model)))
}

# Model 1
# Multiple Linear Regression with only Label, STARS, and AcidIndex as baseline
model1 <- lm(TARGET ~ LabelAppeal_N2 + LabelAppeal_N1 + LabelAppeal_1 +
               LabelAppeal_2 + STARS_NR + STARS_1 + STARS_3 + STARS_4 + 
               AcidIndex,
             data = train_prelim)

model_diag(model1)

# Model 2
# Multiple Linear Regression with backwards selection from the baseline
model2_spec <- lm(TARGET ~ ., data = train_prelim)
model2 <- stepAIC(model2_spec, direction = "backward", trace = 0)

model_diag(model2)

# Model 3
# Poisson regression with backwards selection
model3_spec <- glm(TARGET ~ ., data = train_prelim, 
                   family = poisson(link = "log"))
model3 <- stepAIC(model3_spec, direction = "backward", trace = 0)

model_diag(model3)

# Model 4
# Zero-inflated Poisson regression starting with Model 3 terms
# Manual selection due to fitting issues
# Started with Model 3 terms for both zinf and base parts of models then
# removed terms until it was possible to get convergence
model4 <- zeroinfl(TARGET ~ LabelAppeal_N2 + LabelAppeal_N1 +
                     LabelAppeal_1 + LabelAppeal_2 + STARS_NR + STARS_1 +
                     STARS_3 + STARS_4 + AcidIndex + VolatileAcidity_cf +
                     Alcohol_cf | LabelAppeal_1 + LabelAppeal_2 + STARS_NR +
                     LabelAppeal_N1, 
                   data = train_prelim, 
                   dist = "poisson", link = "log")
model_diag(model4)

# Model 5
# Negative Binomial regression with same terms as Model 3 due to fit issues
model5 <- glm.nb(model3$formula, data = train_prelim)
model_diag(model5)

# Model 6
# Zero-inflated Negative Binomial regression using Model 4 terms due to fit
# issues
model6 <- zeroinfl(TARGET ~ LabelAppeal_N2 + LabelAppeal_N1 +
                     LabelAppeal_1 + LabelAppeal_2 + STARS_NR + STARS_1 +
                     STARS_3 + STARS_4 + AcidIndex + VolatileAcidity_cf +
                     Alcohol_cf | LabelAppeal_1 + LabelAppeal_2 + STARS_NR +
                     LabelAppeal_N1, 
                   data = train_prelim, 
                   dist = "negbin", link = "log")
model_diag(model6)

# Models 7 and 8
# Logistic Regression with backwards selection predicting whether any is sold
model7_spec <- glm(ifelse(TARGET > 0, 1, 0) ~ . -TARGET,
                   data = train_prelim, family = binomial(link = "logit"))
model7 <- stepAIC(model7_spec, direction = "backward", trace = 0)

summary(model7)
vif(model7)

# Poisson regression conditional on being sold
# Subtracts 1 as zero should be handled exclusively by logistic regression
model8_spec <- glm((TARGET - 1) ~ . -TARGET,
                   data = train_prelim[train_prelim$TARGET > 0, ], 
                   family = poisson(link = "log"))
model8 <- stepAIC(model8_spec, direction = "backward", trace = 0)

summary(model8)
vif(model8)

# Training MSE
pred7 <- predict(model7, train_prelim, type = "response")
pred8 <- predict(model8, train_prelim, type = "response") + 1
mse_train <- data.frame(pred7, pred8, train_prelim$TARGET)
names(mse_train) <- c("pred7", "pred8", "TARGET")
mse_train <- mse_train %>%
  mutate(pred = pred7 * pred8,
         sq_error = (TARGET - pred) ^ 2) %>%
  summarize(mse_model = mean(sq_error))
print(mse_train[1, 1])

# Validation MSE
pred7 <- predict(model7, train_validation, type = "response")
pred8 <- predict(model8, train_validation, type = "response") + 1
mse_val <- data.frame(pred7, pred8, train_validation$TARGET)
names(mse_val) <- c("pred7", "pred8", "TARGET")
mse_val <- mse_val %>%
  mutate(pred = pred7 * pred8,
         sq_error = (TARGET - pred) ^ 2) %>%
  summarize(mse_model = mean(sq_error))
print(mse_val[1, 1])

# Model 9
# Decision tree with default options
model9 <- rpart(TARGET ~ ., data = train_prelim)

# Training MSE
predictions <- predict(model9, train_prelim)
mse <- data.frame(predictions, train_prelim$TARGET)
names(mse) <- c("predictions", "TARGET")
mse <- mse %>%
  mutate(sq_error = (TARGET - predictions) ^ 2) %>%
  summarise(mse_model1 = mean(sq_error))
print(mse[1, 1])

# Validation MSE
predictions <- predict(model9, train_validation)
mse <- data.frame(predictions, train_validation$TARGET)
names(mse) <- c("predictions", "TARGET")
mse <- mse %>%
  mutate(sq_error = (TARGET - predictions) ^ 2) %>%
  summarise(mse_model1 = mean(sq_error))
print(mse[1, 1])

# Print coefficients for stand alone program
options(scipen=999)
print(model7$coefficients)
print(model8$coefficients)

# Stand alone program
library(dplyr)

# Load data
training <- read.csv("Wine_Training.csv")
testing <- read.csv("Wine_Test.csv")

# Function to impute a variable with the mean
impute_mean <- function(v, base) {
  ifelse(is.na(v), mean(base, na.rm = TRUE), v)
}

# Function to create missing indicator
miss_ind <- function(v) {
  ifelse(is.na(v), 1, 0)
}

# Function to cap/flr a variable
cap_flr <- function(v, base, mi = 0.025, ma = 0.975) {
  ifelse(v > quantile(base, na.rm = TRUE, ma), quantile(base, na.rm = TRUE, ma),
         ifelse(v < quantile(base, na.rm = TRUE, mi), 
                quantile(base, na.rm = TRUE, mi), v))
}

# Function to create an indicator if capping or flooring is applied
cap_flr_ind <- function(v, base, mi = 0.025, ma = 0.975) {
  ifelse(v > quantile(base, na.rm = TRUE, ma), 1,
         ifelse(v < quantile(base, na.rm = TRUE, mi), 1, 0))
}

# Convert stars to a factor where missing is its own category
# Create a version of LabelAppeal as a factor to compete with the numeric type
# Impute missing with the training mean
# Create missing indicator variables
# Cap and floor most variables at 2.5% and 97.5% of training data
# Create indicators of whether capping or flooring occurred
test_process1 <- testing %>%
  mutate(STARS_NR = ifelse(is.na(STARS), 1, 0),
         STARS_1 = ifelse(STARS == 1 & STARS_NR == 0, 1, 0),
         STARS_3 = ifelse(STARS == 3 & STARS_NR == 0, 1, 0),
         STARS_4 = ifelse(STARS == 4 & STARS_NR == 0, 1, 0),
         LabelAppeal_N2 = ifelse(LabelAppeal == -2, 1, 0),
         LabelAppeal_N1 = ifelse(LabelAppeal == -1, 1, 0),
         LabelAppeal_1 = ifelse(LabelAppeal == 1, 1, 0),
         LabelAppeal_2 = ifelse(LabelAppeal == 2, 1, 0),
         ResidualSugar_imp = impute_mean(ResidualSugar, training$ResidualSugar),
         Chlorides_imp = impute_mean(Chlorides, training$Chlorides),
         FreeSulfurDioxide_imp = impute_mean(FreeSulfurDioxide,
                                             training$FreeSulfurDioxide),
         TotalSulfurDioxide_imp = impute_mean(TotalSulfurDioxide,
                                              training$TotalSulfurDioxide),
         pH_imp = impute_mean(pH, training$pH),
         Sulphates_imp = impute_mean(Sulphates, training$Sulphates),
         Alcohol_imp = impute_mean(Alcohol, training$Alcohol),
         ResidualSugar_mind = miss_ind(ResidualSugar),
         Chlorides_mind = miss_ind(Chlorides),
         FreeSulfurDioxide_mind = miss_ind(FreeSulfurDioxide),
         TotalSulfurDioxide_mind = miss_ind(TotalSulfurDioxide),
         pH_mind = miss_ind(pH),
         Sulphates_mind = miss_ind(Sulphates),
         Alcohol_mind = miss_ind(Alcohol),
         FixedAcidity_cf = cap_flr(FixedAcidity, training$FixedAcidity),
         FixedAcidity_cfind = cap_flr_ind(FixedAcidity, training$FixedAcidity),
         VolatileAcidity_cf = cap_flr(VolatileAcidity,
                                      training$VolatileAcidity),
         VolatileAcidity_cfind = cap_flr_ind(VolatileAcidity, 
                                             training$VolatileAcidity),
         CitricAcid_cf = cap_flr(CitricAcid, training$CitricAcid),
         CitricAcid_cfind = cap_flr_ind(CitricAcid, training$CitricAcid),
         ResidualSugar_cf = cap_flr(ResidualSugar_imp, training$ResidualSugar),
         ResidualSugar_cfind = cap_flr_ind(ResidualSugar_imp, 
                                           training$ResidualSugar),
         Chlorides_cf = cap_flr(Chlorides_imp, training$Chlorides),
         Chlorides_cfind = cap_flr_ind(Chlorides_imp, training$Chlorides),
         FreeSulfurDioxide_cf = cap_flr(FreeSulfurDioxide_imp,
                                        training$FreeSulfurDioxide),
         FreeSulfurDioxide_cfind = cap_flr_ind(FreeSulfurDioxide_imp,
                                               training$FreeSulfurDioxide),
         TotalSulfurDioxide_cf = cap_flr(TotalSulfurDioxide_imp,
                                         training$TotalSulfurDioxide),
         TotalSulfurDioxide_cfind = cap_flr_ind(TotalSulfurDioxide_imp,
                                                training$TotalSulfurDioxide),
         Density_cf = cap_flr(Density, training$Density),
         Density_cfind = cap_flr_ind(Density, training$Density),
         pH_cf = cap_flr(pH_imp, training$pH),
         pH_cfind = cap_flr_ind(pH_imp, training$pH),
         Sulphates_cf = cap_flr(Sulphates_imp, training$Sulphates),
         Sulphates_cfind = cap_flr_ind(Sulphates_imp, training$Sulphates),
         Alcohol_cf = cap_flr(Alcohol_imp, training$Alcohol),
         Alcohol_cfind = cap_flr_ind(Alcohol_imp, training$Alcohol)) %>%
  select(-STARS, -ResidualSugar, -Chlorides, -FreeSulfurDioxide, -pH, -Alcohol,
         -Sulphates, -TotalSulfurDioxide, -FixedAcidity, -VolatileAcidity,
         -CitricAcid, -ResidualSugar_imp, -Chlorides_imp, -Density, -pH_imp,
         -FreeSulfurDioxide_imp, -TotalSulfurDioxide_imp, -Sulphates_imp,
         -Alcohol_imp, -LabelAppeal)

# Score logistic/poisson hurdle model
scores <- test_process1 %>%
  mutate(SCORE_ZERO1 = 7.884710962 +
           AcidIndex * -0.40036707 +
           STARS_NR	*	-4.226471147	+
           STARS_1	*	-2.433375183	+
           STARS_3	*	15.96117705	+
           STARS_4	*	16.17997786	+
           LabelAppeal_N2	*	0.805522786	+
           LabelAppeal_N1	*	0.462007879	+
           LabelAppeal_1	*	-0.565678287	+
           LabelAppeal_2	*	-1.018312267	+
           TotalSulfurDioxide_mind	*	0.250621062	+
           pH_mind	*	-0.416801675	+
           Sulphates_mind	*	-0.243467117	+
           VolatileAcidity_cf	*	-0.200724296	+
           CitricAcid_cf	*	0.074098365	+
           Chlorides_cf	*	-0.25559373	+
           FreeSulfurDioxide_cf	*	0.000441744	+
           TotalSulfurDioxide_cf	*	0.000962701	+
           pH_cf	*	-0.184933172	+
           Sulphates_cf	*	-0.109916549	+
           Alcohol_cf	*	-0.02960423,
         SCORE_ZERO = exp(SCORE_ZERO1) / (1 + exp(SCORE_ZERO1)),
         SCORE_NONZERO = exp(1.12414013 - 
                               0.02100514 * AcidIndex - 
                               0.21424615	* STARS_NR - 
                               0.1367874 * STARS_1 + 
                               0.11145919 * STARS_3	+ 
                               0.23299838 * STARS_4 - 
                               1.04475165 * LabelAppeal_N2 - 
                               0.37842829 * LabelAppeal_N1 + 
                               0.24289664 * LabelAppeal_1 + 
                               0.43442062 * LabelAppeal_2	- 
                               0.01580445 * VolatileAcidity_cf - 
                               0.05079471	* Chlorides_cfind + 
                               0.01019517 * Alcohol_cf) + 1,
         P_TARGET = SCORE_ZERO * SCORE_NONZERO) %>%
  select(INDEX, P_TARGET)

# Save output
write.csv(scores, "kessler_wine.csv", row.names = FALSE)