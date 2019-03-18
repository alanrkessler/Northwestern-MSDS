# Unit 2 Project
# MSDS 411 Section 56
# Alan Kessler

library(ggplot2)
library(gridExtra) # nolint
library(reshape2)
library(MASS)
library(RColorBrewer) # nolint
library(dplyr)
library(caret)
library(ROCR)
library(rpart)
library(rpart.plot)
library(car)

# Setting colors for plots
pal <- brewer.pal(3, "Set1")

# Load data
training <- read.csv("logit_insurance.csv")
testing <- read.csv("logit_insurance_test.csv")

# Section 1: Data Exploration

# Count the number of claims vs. no claims for the binary target
table(training$TARGET_FLAG)

# Visualize the distribution of the continuous model target
dataLine <- training %>%
  filter(TARGET_AMT > 0) %>%
  summarise(y75 = quantile(TARGET_AMT, 0.75),
            y25 = quantile(TARGET_AMT, 0.25),
            x75 = qnorm(0.75), x25 = qnorm(0.25)) %>%
  mutate(slope = (y75 - y25) / (x75 - x25), intercept = y75 - slope * x75)

# Check for normality in target
p1 <- ggplot(training[training$TARGET_AMT > 0, ], aes(sample = TARGET_AMT)) +
  stat_qq(color = pal[1]) +
  geom_abline(data = dataLine, aes(slope = slope, intercept = intercept)) +
  xlab("Theoretical Quantiles") +
  ylab("Sample Quantiles") +
  ggtitle("Target Normal QQ-Plot")

dataLine_log <- training %>%
  filter(TARGET_AMT > 0) %>%
  summarise(y75 = quantile(log(TARGET_AMT), 0.75),
            y25 = quantile(log(TARGET_AMT), 0.25),
            x75 = qnorm(0.75), x25 = qnorm(0.25)) %>%
  mutate(slope = (y75 - y25) / (x75 - x25), intercept = y75 - slope * x75)

# Check for lognormality in target
p2 <- ggplot(training[training$TARGET_AMT > 0, ],
             aes(sample = log(TARGET_AMT))) +
  stat_qq(color = pal[2]) +
  geom_abline(data = dataLine_log, aes(slope = slope, intercept = intercept)) +
  xlab("Theoretical Quantiles") +
  ylab("Sample Quantiles") +
  ggtitle("Target Lognormal QQ-Plot")

grid.arrange(p1, p2, ncol = 2)

# Some initial formating and indicator creation
train_process1 <- training %>%
  mutate(JOB = as.factor(ifelse(JOB == "", "Other", as.character(JOB))),
         OLDCLAIM_NUM = as.numeric(gsub("[^0-9\\.]", "", OLDCLAIM)),
         BLUEBOOK_NUM = as.numeric(gsub("[^0-9\\.]", "", BLUEBOOK)),
         HOME_VAL_NUM = as.numeric(gsub("[^0-9\\.]", "", HOME_VAL)),
         INCOME_NUM = as.numeric(gsub("[^0-9\\.]", "", INCOME)),
         PARENT1_IND = ifelse(PARENT1 == "Yes", 1, 0),
         MARRIED_IND = ifelse(MSTATUS == "z_No", 0, 1),
         MALE_IND = ifelse(SEX == "z_F", 0, 1),
         COMM_USE_IND = ifelse(CAR_USE == "Commercial", 1, 0),
         RED_CAR_IND = ifelse(RED_CAR == "yes", 1, 0),
         REVOKED_IND = ifelse(REVOKED == "Yes", 1, 0),
         RURAL_IND = ifelse(URBANICITY == "z_Highly Rural/ Rural", 1, 0)) %>%
  dplyr::select(-OLDCLAIM, -BLUEBOOK, -HOME_VAL, -INCOME, -PARENT1, -MSTATUS,
                -SEX, -CAR_USE, -RED_CAR, -REVOKED, -URBANICITY, -INDEX)

test_process1 <- testing %>%
  mutate(JOB = as.factor(ifelse(JOB == "", "Other", as.character(JOB))),
         OLDCLAIM_NUM = as.numeric(gsub("[^0-9\\.]", "", OLDCLAIM)),
         BLUEBOOK_NUM = as.numeric(gsub("[^0-9\\.]", "", BLUEBOOK)),
         HOME_VAL_NUM = as.numeric(gsub("[^0-9\\.]", "", HOME_VAL)),
         INCOME_NUM = as.numeric(gsub("[^0-9\\.]", "", INCOME)),
         PARENT1_IND = ifelse(PARENT1 == "Yes", 1, 0),
         MARRIED_IND = ifelse(MSTATUS == "z_No", 0, 1),
         MALE_IND = ifelse(SEX == "z_F", 0, 1),
         COMM_USE_IND = ifelse(CAR_USE == "Commercial", 1, 0),
         RED_CAR_IND = ifelse(RED_CAR == "yes", 1, 0),
         REVOKED_IND = ifelse(REVOKED == "Yes", 1, 0),
         RURAL_IND = ifelse(URBANICITY == "z_Highly Rural/ Rural", 1, 0)) %>%
  dplyr::select(-OLDCLAIM, -BLUEBOOK, -HOME_VAL, -INCOME, -PARENT1, -MSTATUS,
                -SEX, -CAR_USE, -RED_CAR, -REVOKED, -URBANICITY, -INDEX)

# Visualize missing values in the data
MissingCount <- train_process1 %>%
  mutate_all(is.na) %>%
  summarise_all(sum) %>%
  melt() %>%
  mutate(value = value / nrow(training)) %>%
  filter(value > 0)

ggplot(MissingCount, aes(x = variable, weight = value)) +
  geom_bar(color = "black", fill = pal[1]) +
  xlab("Variable (Those with Missing Values)") +
  ylab("Missing Fraction") +
  ggtitle("Missing Values - Training") +
  theme(axis.text.x = element_text(angle = 10, hjust = 1, size = 8))

vars_continuous <- c("AGE", "YOJ", "TRAVTIME", "TIF", "CAR_AGE",
                     "OLDCLAIM_NUM", "BLUEBOOK_NUM", "HOME_VAL_NUM",
                     "INCOME_NUM")

# Merge training and testing for analyzing predictors
df1 <- train_process1 %>%
  select(vars_continuous) %>%
  mutate(Data = "Training")

df2 <- test_process1 %>%
  select(vars_continuous) %>%
  mutate(Data = "Testing")

df <- rbind(df1, df2) %>%
  melt(id.vars = "Data") %>%
  na.omit()

# Box plots for continuous variables
ggplot(df, aes(x = variable, y = value, color = Data)) +
  xlab("Variable") +
  ylab("Value") +
  ggtitle("Box Plots for Continuous Predictor Variables") +
  geom_boxplot() +
  facet_wrap(~variable, scales = "free") +
  theme(strip.text.x = element_blank())

# Histograms for continuous variables
ggplot(df, aes(value, fill = Data)) +
  xlab("Variable Value") +
  ylab("Count") +
  ggtitle("Histograms for Predictor Variables") +
  geom_histogram(bins = 30, color = "black") +
  facet_wrap(~variable, scales = "free") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Training predictor correlation for continuous variables
corr_input <- train_process1 %>%
  select(vars_continuous, TARGET_FLAG, TARGET_AMT) %>%
  na.omit()

corr <- cor(corr_input)
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
  ggtitle("Correlation - Training Data (Continuous Variables)")

# Univariate relationships with the binary target: Continuous
train_melt <- train_process1 %>%
  select(TARGET_FLAG, vars_continuous) %>%
  mutate_at(vars_continuous, ntile, 10) %>%
  melt(., id.vars = "TARGET_FLAG") %>%
  na.omit() %>%
  group_by(variable, value) %>%
  summarise(n = n(),
            p = mean(TARGET_FLAG),
            log_odds = log(p / (1 - p))) %>%
  ungroup() %>%
  mutate(p_plus = p + qnorm(0.975) * sqrt((p * (1 - p)) / n), # nolint
         p_minus = p - qnorm(0.975) * sqrt((p * (1 - p)) / n), # nolint
         lo_plus = log(p_plus / (1 - p_plus)),
         lo_minus = log(p_minus / (1 - p_minus)))

ggplot(train_melt, aes(value, log_odds)) +
  geom_line() +
  geom_line(y = train_melt$lo_plus, color = pal[1]) +
  geom_line(y = train_melt$lo_minus, color = pal[1]) +
  facet_wrap(~variable, scales = "free") +
  ggtitle("Univariate Relationship with Binary Target with 95% CI") +
  xlab("Predictor Variable Deciles") +
  ylab("Claim Log Odds") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 6))

# Univariate relationships with the continuous target: Continuous
train_melt <- train_process1 %>%
  select(TARGET_AMT, vars_continuous) %>%
  filter(TARGET_AMT > 0) %>%
  melt(., id.vars = "TARGET_AMT") %>%
  na.omit()

ggplot(train_melt, aes(x = value, y = log(TARGET_AMT))) +
  geom_point(fill = pal[1], color = "black", shape = 21, size = 1) +
  geom_smooth(method = lm) +
  facet_wrap(~variable, scales = "free") +
  ggtitle("Univariate Relationship with Continuous Target (Log)") +
  xlab("Predictor Variable Value") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Low-cardinality categorical variables that happen to be numeric
vars_categorical1 <- c("KIDSDRIV", "HOMEKIDS", "CLM_FREQ", "MVR_PTS",
                       "PARENT1_IND", "MARRIED_IND", "MALE_IND", "COMM_USE_IND",
                       "RED_CAR_IND", "REVOKED_IND", "RURAL_IND")

# Histograms for training and testing data: Categorical Group 1
df1 <- train_process1 %>%
  select(vars_categorical1) %>%
  mutate(Data = "Training")

df2 <- test_process1 %>%
  select(vars_categorical1) %>%
  mutate(Data = "Testing")

df <- rbind(df1, df2) %>%
  melt(id.vars = "Data") %>%
  na.omit()

ggplot(df, aes(value, fill = Data)) +
  xlab("Variable Value") +
  ylab("Count") +
  ggtitle("Categorical Predictor Variables") +
  geom_bar(color = "black") +
  facet_wrap(~variable, scales = "free") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Non-numeric high-cardinality categorical variables
vars_categorical2 <- c("JOB", "EDUCATION", "CAR_TYPE")

# Histograms for training and testing data: Categorical Group 2
df1 <- train_process1 %>%
  select(vars_categorical2) %>%
  mutate(Data = "Training")

df2 <- test_process1 %>%
  select(vars_categorical2) %>%
  mutate(Data = "Testing")

df <- rbind(df1, df2) %>%
  melt(id.vars = "Data") %>%
  na.omit()

p1 <- ggplot(df[df$variable == "JOB", ], aes(value, fill = Data)) +
  xlab("Job") +
  ylab("Count") +
  ggtitle("Categorical Predictor Variables") +
  geom_bar(color = "black") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

p2 <- ggplot(df[df$variable == "EDUCATION", ], aes(value, fill = Data)) +
  xlab("Education") +
  ylab("Count") +
  geom_bar(color = "black") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6)) +
  theme(legend.position = "none")

p3 <- ggplot(df[df$variable == "CAR_TYPE", ], aes(value, fill = Data)) +
  xlab("Car Type") +
  ylab("Count") +
  geom_bar(color = "black") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6)) +
  theme(legend.position = "none")

grid.arrange(grobs = list(p1, p2, p3), layout_matrix = rbind(c(1, 1), c(2, 3)))

# Univariate relationship with binary target: Categorical Group 1
train_melt <- train_process1 %>%
  select(vars_categorical1, TARGET_FLAG) %>%
  melt(., id.vars = "TARGET_FLAG") %>%
  group_by(variable, value) %>%
  summarise(n = n(),
            p = mean(TARGET_FLAG),
            log_odds = log(p / (1 - p))) %>%
  ungroup() %>%
  mutate(p_plus = p + qnorm(0.975) * sqrt((p * (1 - p)) / n), # nolint
         p_minus = p - qnorm(0.975) * sqrt((p * (1 - p)) / n), # nolint
         lo_plus = log(p_plus / (1 - p_plus)),
         lo_minus = log(p_minus / (1 - p_minus)))

ggplot(data = train_melt, aes(value, log_odds)) +
  geom_line() +
  geom_line(y = train_melt$lo_plus, color = pal[1]) +
  geom_line(y = train_melt$lo_minus, color = pal[1]) +
  facet_wrap(~variable, scales = "free") +
  scale_x_continuous(breaks = seq(0, 13, by = 1)) +
  xlab("Variable Value") +
  ylab("Claim Log Odds") +
  ggtitle("Univariate Relationship with Binary Target with 95% CI") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Univariate relationship with binary target: Categorical Group 2
train_melt <- train_process1 %>%
  select(vars_categorical2, TARGET_FLAG) %>%
  melt(., id.vars = "TARGET_FLAG") %>%
  group_by(variable, value) %>%
  summarise(n = n(),
            p = mean(TARGET_FLAG),
            log_odds = log(p / (1 - p))) %>%
  ungroup() %>%
  mutate(p_plus = p + qnorm(0.975) * sqrt((p * (1 - p)) / n), #nolint
         p_minus = p - qnorm(0.975) * sqrt((p * (1 - p)) / n), #nolint
         lo_plus = log(p_plus / (1 - p_plus)),
         lo_minus = log(p_minus / (1 - p_minus)))

p1 <- ggplot(data = train_melt[train_melt$variable == "JOB", ],
             aes(factor(value), log_odds, group = 1)) +
  geom_line() +
  geom_line(y = train_melt[train_melt$variable == "JOB", ]$lo_plus,
            color = pal[1], group = 1) +
  geom_line(y = train_melt[train_melt$variable == "JOB", ]$lo_minus,
            color = pal[1], group = 1) +
  xlab("Job") +
  ylab("Claim Log Odds") +
  ggtitle("Univariate Relationship with Binary Target with 95% CI") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

p2 <- ggplot(data = train_melt[train_melt$variable == "EDUCATION", ],
             aes(factor(value), log_odds, group = 1)) +
  geom_line() +
  geom_line(y = train_melt[train_melt$variable == "EDUCATION", ]$lo_plus,
            color = pal[1], group = 1) +
  geom_line(y = train_melt[train_melt$variable == "EDUCATION", ]$lo_minus,
            color = pal[1], group = 1) +
  xlab("Education") +
  ylab("Claim Log Odds") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

p3 <- ggplot(data = train_melt[train_melt$variable == "CAR_TYPE", ],
             aes(factor(value), log_odds, group = 1)) +
  geom_line() +
  geom_line(y = train_melt[train_melt$variable == "CAR_TYPE", ]$lo_plus,
            color = pal[1], group = 1) +
  geom_line(y = train_melt[train_melt$variable == "CAR_TYPE", ]$lo_minus,
            color = pal[1], group = 1) +
  xlab("Car Type") +
  ylab("Claim Log Odds") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

grid.arrange(grobs = list(p1, p2, p3), layout_matrix = rbind(c(1, 1), c(2, 3)))

# Univariate relationship with continuous target: Categorical Group 1
train_melt <- train_process1 %>%
  select(vars_categorical1, TARGET_AMT) %>%
  filter(TARGET_AMT > 0) %>%
  mutate(TARGET_AMT = log(TARGET_AMT)) %>%
  melt(., id.vars = "TARGET_AMT") %>%
  group_by(variable, value) %>%
  summarise(n = n(),
            m = mean(TARGET_AMT),
            v = var(TARGET_AMT)) %>%
  ungroup() %>%
  mutate(m_plus = m + qnorm(0.975) * sqrt(v / n),
         m_minus = m - qnorm(0.975) * sqrt(v / n))

ggplot(data = train_melt, aes(value, m)) +
  geom_line() +
  geom_line(y = train_melt$m_plus, color = pal[1]) +
  geom_line(y = train_melt$m_minus, color = pal[1]) +
  facet_wrap(~variable, scales = "free") +
  scale_x_continuous(breaks = seq(0, 13, by = 1)) +
  xlab("Variable Value") +
  ylab("Average Claim Amount") +
  ggtitle("Training Claim Amount (Log) with 95% CI") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Univariate relationship with continuous target: Categorical Group 2
train_melt <- train_process1 %>%
  select(vars_categorical2, TARGET_AMT) %>%
  filter(TARGET_AMT > 0) %>%
  mutate(TARGET_AMT = log(TARGET_AMT)) %>%
  melt(., id.vars = "TARGET_AMT") %>%
  group_by(variable, value) %>%
  summarise(n = n(),
            m = mean(TARGET_AMT),
            v = var(TARGET_AMT)) %>%
  ungroup() %>%
  mutate(m_plus = m + qnorm(0.975) * sqrt(v / n),
         m_minus = m - qnorm(0.975) * sqrt(v / n))

p1 <- ggplot(data = train_melt[train_melt$variable == "JOB", ],
             aes(factor(value), m, group = 1)) +
  geom_line() +
  geom_line(y = train_melt[train_melt$variable == "JOB", ]$m_plus,
            color = pal[1], group = 1) +
  geom_line(y = train_melt[train_melt$variable == "JOB", ]$m_minus,
            color = pal[1], group = 1) +
  xlab("Job") +
  ylab("Average Claim Amount") +
  ggtitle("Training Claim Amount (Log) with 95% CI") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

p2 <- ggplot(data = train_melt[train_melt$variable == "EDUCATION", ],
             aes(factor(value), m, group = 1)) +
  geom_line() +
  geom_line(y = train_melt[train_melt$variable == "EDUCATION", ]$m_plus,
            color = pal[1], group = 1) +
  geom_line(y = train_melt[train_melt$variable == "EDUCATION", ]$m_minus,
            color = pal[1], group = 1) +
  xlab("Education") +
  ylab("Average Claim Amount") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

p3 <- ggplot(data = train_melt[train_melt$variable == "CAR_TYPE", ],
             aes(factor(value), m, group = 1)) +
  geom_line() +
  geom_line(y = train_melt[train_melt$variable == "CAR_TYPE", ]$m_plus,
            color = pal[1], group = 1) +
  geom_line(y = train_melt[train_melt$variable == "CAR_TYPE", ]$m_minus,
            color = pal[1], group = 1) +
  xlab("Car Type") +
  ylab("Average Claim Amount") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

grid.arrange(grobs = list(p1, p2, p3), layout_matrix = rbind(c(1, 1), c(2, 3)))

# Part 2: Data Preparation

# Drop red car due to small effect and lack of causality
# Impute missing with average
# Create missing indicators (except for age - low sample size)
# Floor car age to 1
# Create zero indicators for continuous variables that can have meaning
# Cap travel time at 65, MVR points at 7, and YOJ at 15
# Cap income, home value, and bluebook value at 98th percentile
# Take log of bluebook value to linearize relationship
# Create indicator that bluebook value was floored
# Create equal sized age bins for likelihood encoding
# Group van and panel truck due to similar log odds
train_process2 <- train_process1 %>%
  select(-RED_CAR_IND) %>%
  mutate(AGE_IMPUTE = ifelse(is.na(AGE),
                             mean(train_process1$AGE, na.rm = TRUE),
                             AGE),
         YOJ_IMPUTE = ifelse(is.na(YOJ),
                             mean(train_process1$YOJ, na.rm = TRUE),
                             YOJ),
         CAR_AGE_IMPUTE = ifelse(is.na(CAR_AGE),
                                 mean(train_process1$CAR_AGE, na.rm = TRUE),
                                 CAR_AGE),
         HOME_VAL_IMPUTE = ifelse(is.na(HOME_VAL_NUM),
                                  mean(train_process1$HOME_VAL_NUM,
                                       na.rm = TRUE),
                                  HOME_VAL_NUM),
         INCOME_IMPUTE = ifelse(is.na(INCOME_NUM),
                                mean(train_process1$INCOME_NUM, na.rm = TRUE),
                                INCOME_NUM),
         YOJ_MISS_IND = ifelse(is.na(YOJ), 1, 0),
         CAR_AGE_MISS_IND = ifelse(is.na(CAR_AGE), 1, 0),
         HOME_VAL_MISS_IND = ifelse(is.na(HOME_VAL_NUM), 1, 0),
         INCOME_MISS_IND = ifelse(is.na(INCOME_NUM), 1, 0),
         CAR_AGE_FLR = ifelse(CAR_AGE_IMPUTE < 1, 1, CAR_AGE_IMPUTE),
         TRAVTIME_FIVE_IND = ifelse(TRAVTIME == 5, 1, 0),
         TIF_ONE_IND = ifelse(TIF == 1, 1, 0),
         CAR_AGE_ONE_IND = ifelse(CAR_AGE_FLR == 1, 1, 0),
         HOME_VAL_ZERO_IND = ifelse(HOME_VAL_IMPUTE == 0, 1, 0),
         INCOME_ZERO_IND = ifelse(INCOME_IMPUTE == 0, 1, 0),
         YOJ_CAP = ifelse(YOJ_IMPUTE > 15, 15, YOJ_IMPUTE),
         TRAVTIME_CAP = ifelse(TRAVTIME > 65, 65, TRAVTIME),
         MVR_PTS_CAP = ifelse(MVR_PTS > 7, 7, MVR_PTS),
         INCOME_CAP = ifelse(INCOME_IMPUTE > quantile(INCOME_NUM, 0.98,
                                                      na.rm = TRUE),
                             quantile(INCOME_NUM, 0.98, na.rm = TRUE),
                             INCOME_IMPUTE),
         HOME_VAL_CAP = ifelse(HOME_VAL_IMPUTE > quantile(HOME_VAL_NUM, 0.98,
                                                          na.rm = TRUE),
                               quantile(HOME_VAL_NUM, 0.98, na.rm = TRUE),
                               HOME_VAL_IMPUTE),
         BLUEBOOK_CLOG = log(ifelse(BLUEBOOK_NUM > quantile(BLUEBOOK_NUM, 0.98,
                                                            na.rm = TRUE),
                                    quantile(BLUEBOOK_NUM, 0.98, na.rm = TRUE),
                                    BLUEBOOK_NUM)),
         BLUEBOOK_MIN = ifelse(BLUEBOOK_NUM == 1500, 1, 0),
         OLDCLAIM_CLOG = log1p(ifelse(OLDCLAIM_NUM > quantile(OLDCLAIM_NUM,
                                                              0.98,
                                                              na.rm = TRUE),
                                      quantile(OLDCLAIM_NUM, 0.98,
                                               na.rm = TRUE),
                                      OLDCLAIM_NUM)),
         AGE_BIN = ntile(AGE_IMPUTE, 10),
         KIDSDRIV_IND = ifelse(KIDSDRIV > 1, 1, KIDSDRIV),
         HOMEKIDS_IND = ifelse(HOMEKIDS > 1, 1, HOMEKIDS),
         CLM_FREQ_IND = ifelse(CLM_FREQ > 1, 1, CLM_FREQ),
         CAR_TYPE_GRP = ifelse(as.character(CAR_TYPE) == "Panel Truck",
                               "PT_Van",
                               ifelse(as.character(CAR_TYPE) == "Van",
                                      "PT_Van",
                                      as.character(CAR_TYPE)))) %>%
  select(-AGE, -YOJ, -CAR_AGE, -HOME_VAL_NUM, -INCOME_NUM, -CAR_AGE_IMPUTE,
         -TRAVTIME, -INCOME_IMPUTE, -HOME_VAL_IMPUTE, -BLUEBOOK_NUM,
         -KIDSDRIV, -HOMEKIDS, -CLM_FREQ, -CAR_TYPE, -MVR_PTS, -YOJ_IMPUTE,
         -OLDCLAIM_NUM)

# Create factor version of the binary target for use with caret package
train_process2$TARGET_FACTOR <- factor(ifelse(train_process2$TARGET_FLAG == 1,
                                              "Claim", "No_Claim"),
                                       levels = c("No_Claim", "Claim"))

# Create likelihood encodings for age deciles to reflect unique relationship
# Also preserves monotonicity 
train_age_bins <- train_process2 %>%
  select(AGE_IMPUTE, TARGET_FLAG) %>%
  mutate(AGE_BIN = ntile(AGE_IMPUTE, 10)) %>%
  group_by(AGE_BIN) %>%
  summarise(AGE_LE = mean(TARGET_FLAG))

train_education <- train_process2 %>%
  select(EDUCATION, TARGET_FLAG) %>%
  group_by(EDUCATION) %>%
  summarise(EDUCATION_LE = mean(TARGET_FLAG))

train_job <- train_process2 %>%
  select(JOB, TARGET_FLAG) %>%
  group_by(JOB) %>%
  summarise(JOB_LE = mean(TARGET_FLAG))

train_car_type <- train_process2 %>%
  select(CAR_TYPE_GRP, TARGET_FLAG) %>%
  group_by(CAR_TYPE_GRP) %>%
  summarise(CAR_TYPE_LE = mean(TARGET_FLAG))

train_process3 <- train_process2 %>%
  left_join(., train_age_bins, by = c("AGE_BIN")) %>%
  left_join(., train_education, by = c("EDUCATION")) %>%
  left_join(., train_job, by = c("JOB")) %>%
  left_join(., train_car_type, by = c("CAR_TYPE_GRP")) %>%
  select(-AGE_BIN)

vars_continuous <- c("YOJ_CAP", "TRAVTIME_CAP", "TIF", "MVR_PTS_CAP",
                     "CAR_AGE_FLR", "OLDCLAIM_CLOG", "BLUEBOOK_CLOG",
                     "HOME_VAL_CAP", "INCOME_CAP")

# Univariate relationships with the binary target: Continuous
train_melt <- train_process3 %>%
  select(TARGET_FLAG, vars_continuous) %>%
  mutate_at(vars_continuous, ntile, 10) %>%
  melt(., id.vars = "TARGET_FLAG") %>%
  na.omit() %>%
  group_by(variable, value) %>%
  summarise(n = n(),
            p = mean(TARGET_FLAG),
            log_odds = log(p / (1 - p))) %>%
  ungroup() %>%
  mutate(p_plus = p + qnorm(0.975) * sqrt((p * (1 - p)) / n), # nolint
         p_minus = p - qnorm(0.975) * sqrt((p * (1 - p)) / n), # nolint
         lo_plus = log(p_plus / (1 - p_plus)),
         lo_minus = log(p_minus / (1 - p_minus)))

ggplot(train_melt, aes(value, log_odds)) +
  geom_line() +
  geom_line(y = train_melt$lo_plus, color = pal[1]) +
  geom_line(y = train_melt$lo_minus, color = pal[1]) +
  facet_wrap(~variable, scales = "free") +
  ggtitle("Univariate Relationship with Binary Target with 95% CI") +
  xlab("Predictor Variable Deciles") +
  ylab("Claim Log Odds") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 6))

# Plot categorical final variables
vars_categorical1 <- c("KIDSDRIV_IND", "HOMEKIDS_IND", "CLM_FREQ_IND",
                       "PARENT1_IND", "MARRIED_IND", "MALE_IND", "COMM_USE_IND",
                       "REVOKED_IND", "RURAL_IND", "YOJ_MISS_IND",
                       "CAR_AGE_MISS_IND", "HOME_VAL_MISS_IND")
vars_categorical2 <- c("INCOME_MISS_IND", "TRAVTIME_FIVE_IND",
                       "TIF_ONE_IND", "CAR_AGE_ONE_IND",
                       "HOME_VAL_ZERO_IND", "BLUEBOOK_MIN", "YOJ_MISS_IND",
                       "CAR_AGE_MISS_IND", "HOME_VAL_MISS_IND")

train_melt <- train_process3 %>%
  select(vars_categorical1, TARGET_FLAG) %>%
  melt(., id.vars = "TARGET_FLAG") %>%
  group_by(variable, value) %>%
  summarise(n = n(),
            p = mean(TARGET_FLAG),
            log_odds = log(p / (1 - p))) %>%
  ungroup() %>%
  mutate(p_plus = p + qnorm(0.975) * sqrt((p * (1 - p)) / n), # nolint
         p_minus = p - qnorm(0.975) * sqrt((p * (1 - p)) / n), # nolint
         lo_plus = log(p_plus / (1 - p_plus)),
         lo_minus = log(p_minus / (1 - p_minus)))

ggplot(data = train_melt, aes(value, log_odds)) +
  geom_line(group = 1) +
  geom_line(y = train_melt$lo_plus, color = pal[1]) +
  geom_line(y = train_melt$lo_minus, color = pal[1]) +
  facet_wrap(~variable, scales = "free") +
  scale_x_continuous(breaks = seq(0, 13, by = 1)) +
  xlab("Variable Value") +
  ylab("Claim Log Odds") +
  ggtitle("Univariate Relationship with Binary Target with 95% CI") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

train_melt <- train_process3 %>%
  select(vars_categorical2, TARGET_FLAG) %>%
  melt(., id.vars = "TARGET_FLAG") %>%
  group_by(variable, value) %>%
  summarise(n = n(),
            p = mean(TARGET_FLAG),
            log_odds = log(p / (1 - p))) %>%
  ungroup() %>%
  mutate(p_plus = p + qnorm(0.975) * sqrt((p * (1 - p)) / n), # nolint
         p_minus = p - qnorm(0.975) * sqrt((p * (1 - p)) / n), # nolint
         lo_plus = log(p_plus / (1 - p_plus)),
         lo_minus = log(p_minus / (1 - p_minus)))

ggplot(data = train_melt, aes(value, log_odds)) +
  geom_line(group = 1) +
  geom_line(y = train_melt$lo_plus, color = pal[1]) +
  geom_line(y = train_melt$lo_minus, color = pal[1]) +
  facet_wrap(~variable, scales = "free") +
  scale_x_continuous(breaks = seq(0, 13, by = 1)) +
  xlab("Variable Value") +
  ylab("Claim Log Odds") +
  ggtitle("Univariate Relationship with Binary Target with 95% CI") +
  theme(strip.text.x = element_text(size = 7),
        axis.text.x = element_text(size = 6),
        axis.text.y = element_text(size = 6))

# Training final predictor correlation (not great for size - zoom in)
corr_input <- train_process3 %>%
  select(-TARGET_FACTOR, -EDUCATION, -JOB, -CAR_TYPE_GRP, -AGE_IMPUTE) %>%
  na.omit()

corr <- cor(corr_input)
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
  ggtitle("Correlation - Training Data (Final Predictors)")

# Part 3: Build Models

# Create standardized output function for all models
metric_printout <- function(model_object) {
  par(mfrow = c(1, 1))

  # Print final model summary
  print(summary(model_object))

  # Cross-Validation (Resampled) Metrics
  # ROCR requires numeric labels
  cv_scores <- model_object$pred %>%
    mutate(actual = ifelse(obs == "Claim", 1, 0))

  # KS statistic
  pred <- prediction(model_object$pred$Claim, cv_scores$actual)
  perf <- performance(pred, "tpr", "fpr")
  ks <- max(perf@y.values[[1]] - perf@x.values[[1]])

  # ROC curve
  plot(perf, main = "ROC Curve (Resampled)")
  lines(x = c(0, 1), y = c(0, 1), lty = 2)

  # AUC
  auc <- performance(pred, measure = "auc")@y.values[[1]]

  # Training Data Metrics
  # Format for ROCR
  scores <- data.frame(model_object$finalModel$fitted.values) # nolint

  names(scores) <- c("Claim")

  scores <- scores %>%
    cbind(., model_object$finalModel$data) %>% # nolint
    rename(obs = .outcome) %>%
    mutate(actual = ifelse(obs == "Claim", 1, 0))

  # KS statistic
  pred <- prediction(scores$Claim, scores$actual)
  perf <- performance(pred, "tpr", "fpr")
  kst <- max(perf@y.values[[1]] - perf@x.values[[1]])

  # ROC curve
  plot(perf, main = "ROC Curve (Training)")
  lines(x = c(0, 1), y = c(0, 1), lty = 2)

  # AUC
  auct <- performance(pred, measure = "auc")@y.values[[1]]

  # Log odds diagnostic plot (should show linearity)
  log_odds <- cv_scores %>%
    mutate(Claim_bucket = ntile(Claim, 50)) %>%
    group_by(Claim_bucket) %>%
    summarise(cv_pred = mean(Claim),
              actual_pred = mean(actual)) %>%
    ungroup() %>%
    mutate(cv_lo = log(cv_pred / (1 - cv_pred)),
           actual_lo = log(actual_pred / (1 - actual_pred)))

  plot(actual_lo ~ cv_lo, data = log_odds,
       xlab = "Log Odds - Estimated", ylab = "Log Odds - Actual",
       main = "Cross Validation Log-Odds (50 Groups)")
  abline(0, 1)

  # Lift chart to observe ranking
  lift <- cv_scores %>%
    mutate(Claim_bucket = ntile(Claim, 10)) %>%
    group_by(Claim_bucket) %>%
    summarise(cv_pred = mean(Claim),
              actual_pred = mean(actual)) %>%
    ungroup()

  lift_plot <- ggplot(data = lift, aes(x = Claim_bucket, y = actual_pred)) +
    geom_bar(stat = "identity", color = "black", fill = pal[2]) +
    scale_x_continuous(name = "Prediction Decile", breaks = seq(1, 10, 1)) +
    ylab("Out of Fold Actual Empirical Probability") +
    ggtitle("Cross Validation Lift Chart")

  print(lift_plot)

  # Print Metrics
  print(paste0("AUC (Resampled): ", auc))
  print(paste0("AUC (Training): ", auct))
  print(paste0("KS Stat (Resampled): ", ks))
  print(paste0("KS Stat (Training): ", kst))
  print(paste0("Lift Top to Bottom Ratio: ", lift[10, 2] / lift[1, 2]))

}

# Models will be assessed based on five-fold cross validation
set.seed(827)
fit_control <- trainControl(method = "cv", number = 5,
                            savePredictions = TRUE, classProbs = TRUE,
                            summaryFunction = twoClassSummary)

# Models 1-5: Binary Target
# Model 1: Simple Model (Prior Claim or Not)
glm_fit1 <- train(TARGET_FACTOR ~ CLM_FREQ_IND, data = train_process3,
                  method = "glm", family = binomial(link = "logit"),
                  trControl = fit_control, metric = "ROC")

metric_printout(glm_fit1)

# Model 2: Forwards Selection
glm_fit2 <- train(TARGET_FACTOR ~ TIF + OLDCLAIM_CLOG + PARENT1_IND +
                    MARRIED_IND + MALE_IND + COMM_USE_IND + REVOKED_IND +
                    RURAL_IND + YOJ_MISS_IND + CAR_AGE_MISS_IND +
                    HOME_VAL_MISS_IND + INCOME_MISS_IND + CAR_AGE_FLR +
                    TRAVTIME_FIVE_IND + TIF_ONE_IND + CAR_AGE_ONE_IND +
                    HOME_VAL_ZERO_IND + INCOME_ZERO_IND + YOJ_CAP +
                    TRAVTIME_CAP + MVR_PTS_CAP + INCOME_CAP + HOME_VAL_CAP +
                    BLUEBOOK_CLOG + BLUEBOOK_MIN + KIDSDRIV_IND + HOMEKIDS_IND +
                    CLM_FREQ_IND + AGE_LE + EDUCATION_LE + JOB_LE + CAR_TYPE_LE,
                  data = train_process3,
                  method = "glmStepAIC",
                  direction = "forward",
                  trace = 0,
                  family = binomial(link = "logit"),
                  trControl = fit_control,
                  metric = "ROC")

metric_printout(glm_fit2)
vif(glm_fit2$finalModel) # nolint

# Model 3: Model 2 but Probit rather than logit
glm_fit3 <- train(TARGET_FACTOR ~ TIF + OLDCLAIM_CLOG + PARENT1_IND +
                    MARRIED_IND + MALE_IND + COMM_USE_IND + REVOKED_IND +
                    RURAL_IND + YOJ_MISS_IND + CAR_AGE_MISS_IND +
                    HOME_VAL_MISS_IND + INCOME_MISS_IND + CAR_AGE_FLR +
                    TRAVTIME_FIVE_IND + TIF_ONE_IND + CAR_AGE_ONE_IND +
                    HOME_VAL_ZERO_IND + INCOME_ZERO_IND + YOJ_CAP +
                    TRAVTIME_CAP + MVR_PTS_CAP + INCOME_CAP + HOME_VAL_CAP +
                    BLUEBOOK_CLOG + BLUEBOOK_MIN + KIDSDRIV_IND + HOMEKIDS_IND +
                    CLM_FREQ_IND + AGE_LE + EDUCATION_LE + JOB_LE + CAR_TYPE_LE,
                  data = train_process3,
                  method = "glmStepAIC",
                  direction = "forward",
                  trace = 0,
                  family = binomial(link = "probit"),
                  trControl = fit_control,
                  metric = "ROC")

metric_printout(glm_fit3)

# Generate tree to evaluate interactions (modify controls to adjust)
# Based on selected main effects from Model 2
# Dropped old_claim_clog due to large VIF
tree <- rpart(TARGET_FACTOR ~ TIF + PARENT1_IND + MARRIED_IND +
                COMM_USE_IND + REVOKED_IND + RURAL_IND + INCOME_ZERO_IND +
                YOJ_CAP + TRAVTIME_CAP + MVR_PTS_CAP + INCOME_CAP +
                HOME_VAL_CAP + BLUEBOOK_CLOG + KIDSDRIV_IND + CLM_FREQ_IND +
                AGE_LE + EDUCATION_LE + JOB_LE + CAR_TYPE_LE,
              data = train_process3, control = rpart.control(cp = 0.005,
                                                             maxdepth = 8))
rpart.plot(tree)

# Model 4: Additional Interactions
# Specify base model for interactions
# Only interactions containing main effects from model 2 are considered
glm4_spec <- glm(TARGET_FLAG ~ TIF + PARENT1_IND +
                   MARRIED_IND + COMM_USE_IND + REVOKED_IND + RURAL_IND +
                   INCOME_ZERO_IND + YOJ_CAP + TRAVTIME_CAP + MVR_PTS_CAP +
                   INCOME_CAP + HOME_VAL_CAP + BLUEBOOK_CLOG + KIDSDRIV_IND +
                   CLM_FREQ_IND + AGE_LE + EDUCATION_LE + JOB_LE +
                   CAR_TYPE_LE,
                 data = train_process3,
                 family = binomial(link = "logit"))

# Evaluate interactions (only first 5 to come into model)
glm_fit4 <- stepAIC(glm4_spec, direction = "forward",
                    scope = list(lower = glm4_spec, upper = ~ . ^ 2),
                    trace = 0,
                    steps = 7)

# Print formula to use in model 5
glm_fit4$formula

# Fit the model in the five fold cross validation context
glm_fit4c <- train(TARGET_FACTOR ~ TIF + PARENT1_IND + MARRIED_IND +
                     COMM_USE_IND + REVOKED_IND + RURAL_IND + INCOME_ZERO_IND +
                     YOJ_CAP + TRAVTIME_CAP + MVR_PTS_CAP + INCOME_CAP +
                     HOME_VAL_CAP + BLUEBOOK_CLOG + KIDSDRIV_IND +
                     CLM_FREQ_IND + AGE_LE + EDUCATION_LE + JOB_LE +
                     CAR_TYPE_LE + KIDSDRIV_IND:JOB_LE +
                     MVR_PTS_CAP:CLM_FREQ_IND + TRAVTIME_CAP:JOB_LE +
                     RURAL_IND:CLM_FREQ_IND + RURAL_IND:JOB_LE +
                     RURAL_IND:CAR_TYPE_LE + INCOME_CAP:KIDSDRIV_IND,
                   data = train_process3,
                   method = "glm",
                   family = binomial(link = "logit"),
                   trControl = fit_control,
                   metric = "ROC")

metric_printout(glm_fit4c)
# Check to see if model performs similarly across folds
glm_fit4c$resample

# Change control function for severity models
fit_control <- trainControl(method = "cv", number = 5,
                            savePredictions = TRUE)

# Models 5-6: Continuous Target
# Model 5: OLS Regression
glm_fit5 <- train(TARGET_AMT ~ TIF + OLDCLAIM_CLOG + PARENT1_IND +
                    MARRIED_IND + MALE_IND + COMM_USE_IND + REVOKED_IND +
                    RURAL_IND + YOJ_MISS_IND + CAR_AGE_MISS_IND +
                    HOME_VAL_MISS_IND + INCOME_MISS_IND + CAR_AGE_FLR +
                    TRAVTIME_FIVE_IND + TIF_ONE_IND + CAR_AGE_ONE_IND +
                    HOME_VAL_ZERO_IND + INCOME_ZERO_IND + YOJ_CAP +
                    TRAVTIME_CAP + MVR_PTS_CAP + INCOME_CAP + HOME_VAL_CAP +
                    BLUEBOOK_CLOG + BLUEBOOK_MIN + KIDSDRIV_IND + HOMEKIDS_IND +
                    CLM_FREQ_IND + AGE_IMPUTE + EDUCATION + JOB + CAR_TYPE_GRP +
                    AGE_IMPUTE ^ 2,
                  data = train_process3[train_process3$TARGET_AMT > 0, ],
                  method = "glmStepAIC",
                  direction = "backward",
                  trace = 0,
                  family = gaussian(link = "identity"),
                  trControl = fit_control)

glm_fit5
summary(glm_fit5)
par(mfrow = c(2, 2))
plot(glm_fit5$finalModel) # nolint
vif(glm_fit5$finalModel) # nolint

# Lift chart to observe ranking
lift <- glm_fit5$pred %>%
  mutate(pred_bucket = ntile(pred, 10)) %>%
  group_by(pred_bucket) %>%
  summarise(actual_pred = mean(obs)) %>%
  ungroup()

lift_plot <- ggplot(data = lift, aes(x = pred_bucket, y = actual_pred)) +
  geom_bar(stat = "identity", color = "black", fill = pal[2]) +
  scale_x_continuous(name = "Prediction Decile", breaks = seq(1, 10, 1)) +
  ylab("Out of Fold Actual Average Claim Severity") +
  ggtitle("Cross Validation Lift Chart")

lift_plot

print(paste0("Lift Top to Bottom Ratio: ", lift[10, 2] / lift[1, 2]))

# Model 6: Gamma GLM
glm_fit6 <- train(TARGET_AMT ~ TIF + OLDCLAIM_CLOG + PARENT1_IND +
                    MARRIED_IND + MALE_IND + COMM_USE_IND + REVOKED_IND +
                    RURAL_IND + YOJ_MISS_IND + CAR_AGE_MISS_IND +
                    HOME_VAL_MISS_IND + INCOME_MISS_IND + CAR_AGE_FLR +
                    TRAVTIME_FIVE_IND + TIF_ONE_IND + CAR_AGE_ONE_IND +
                    HOME_VAL_ZERO_IND + INCOME_ZERO_IND + YOJ_CAP +
                    TRAVTIME_CAP + MVR_PTS_CAP + INCOME_CAP + HOME_VAL_CAP +
                    BLUEBOOK_CLOG + BLUEBOOK_MIN + KIDSDRIV_IND + HOMEKIDS_IND +
                    CLM_FREQ_IND + AGE_IMPUTE + EDUCATION + JOB + CAR_TYPE_GRP +
                    AGE_IMPUTE ^ 2,
                  data = train_process3[train_process3$TARGET_AMT > 0, ],
                  method = "glmStepAIC",
                  direction = "backward",
                  trace = 0,
                  family = Gamma(link = "log"),
                  trControl = fit_control)
glm_fit6
summary(glm_fit6)
plot(glm_fit6$finalModel) # nolint
vif(glm_fit6$finalModel) # nolint

# Lift chart to observe ranking
lift <- glm_fit6$pred %>%
  mutate(pred_bucket = ntile(pred, 10)) %>%
  group_by(pred_bucket) %>%
  summarise(actual_pred = mean(obs)) %>%
  ungroup()

lift_plot <- ggplot(data = lift, aes(x = pred_bucket, y = actual_pred)) +
  geom_bar(stat = "identity", color = "black", fill = pal[2]) +
  scale_x_continuous(name = "Prediction Decile", breaks = seq(1, 10, 1)) +
  ylab("Out of Fold Actual Average Claim Severity") +
  ggtitle("Cross Validation Lift Chart")

lift_plot

print(paste0("Lift Top to Bottom Ratio: ", lift[10, 2] / lift[1, 2]))

# Fit model two without prior claim cost
fit_control <- trainControl(method = "cv", number = 5,
                            savePredictions = TRUE, classProbs = TRUE,
                            summaryFunction = twoClassSummary)

# Fit the model in the five fold cross validation context
glm_fit2c <- train(TARGET_FACTOR ~ TIF + PARENT1_IND + MARRIED_IND +
                     COMM_USE_IND + REVOKED_IND + RURAL_IND + INCOME_ZERO_IND +
                     YOJ_CAP + TRAVTIME_CAP + MVR_PTS_CAP + INCOME_CAP +
                     HOME_VAL_CAP + BLUEBOOK_CLOG + KIDSDRIV_IND +
                     CLM_FREQ_IND + AGE_LE + EDUCATION_LE + JOB_LE +
                     CAR_TYPE_LE,
                   data = train_process3,
                   method = "glm",
                   family = binomial(link = "logit"),
                   trControl = fit_control,
                   metric = "ROC")

metric_printout(glm_fit2c)
# Check to see if model performs similarly across folds
glm_fit2c$resample

# Print inputs for deployment code
options(scipen = 999)

# Model 2 Coefficients
glm_fit2c$finalModel$coefficients # nolint

# Model 6 Coefficients
glm_fit6$finalModel$coefficients # nolint

# Stand alone scoring program

library(dplyr)

# Load data
training <- read.csv("logit_insurance.csv")
testing <- read.csv("logit_insurance_test.csv")

# Initial formating and indicator creation
train_process1 <- training %>%
  mutate(JOB = as.factor(ifelse(JOB == "", "Other", as.character(JOB))),
         OLDCLAIM_NUM = as.numeric(gsub("[^0-9\\.]", "", OLDCLAIM)),
         BLUEBOOK_NUM = as.numeric(gsub("[^0-9\\.]", "", BLUEBOOK)),
         HOME_VAL_NUM = as.numeric(gsub("[^0-9\\.]", "", HOME_VAL)),
         INCOME_NUM = as.numeric(gsub("[^0-9\\.]", "", INCOME)),
         PARENT1_IND = ifelse(PARENT1 == "Yes", 1, 0),
         MARRIED_IND = ifelse(MSTATUS == "z_No", 0, 1),
         MALE_IND = ifelse(SEX == "z_F", 0, 1),
         COMM_USE_IND = ifelse(CAR_USE == "Commercial", 1, 0),
         REVOKED_IND = ifelse(REVOKED == "Yes", 1, 0),
         RURAL_IND = ifelse(URBANICITY == "z_Highly Rural/ Rural", 1, 0)) %>%
  dplyr::select(-OLDCLAIM, -BLUEBOOK, -HOME_VAL, -INCOME, -PARENT1, -MSTATUS,
                -SEX, -CAR_USE, -RED_CAR, -REVOKED, -URBANICITY, -INDEX)

test_process1 <- testing %>%
  mutate(JOB = as.factor(ifelse(JOB == "", "Other", as.character(JOB))),
         OLDCLAIM_NUM = as.numeric(gsub("[^0-9\\.]", "", OLDCLAIM)),
         BLUEBOOK_NUM = as.numeric(gsub("[^0-9\\.]", "", BLUEBOOK)),
         HOME_VAL_NUM = as.numeric(gsub("[^0-9\\.]", "", HOME_VAL)),
         INCOME_NUM = as.numeric(gsub("[^0-9\\.]", "", INCOME)),
         PARENT1_IND = ifelse(PARENT1 == "Yes", 1, 0),
         MARRIED_IND = ifelse(MSTATUS == "z_No", 0, 1),
         MALE_IND = ifelse(SEX == "z_F", 0, 1),
         COMM_USE_IND = ifelse(CAR_USE == "Commercial", 1, 0),
         REVOKED_IND = ifelse(REVOKED == "Yes", 1, 0),
         RURAL_IND = ifelse(URBANICITY == "z_Highly Rural/ Rural", 1, 0)) %>%
  dplyr::select(-OLDCLAIM, -BLUEBOOK, -HOME_VAL, -INCOME, -PARENT1, -MSTATUS,
                -SEX, -CAR_USE, -RED_CAR, -REVOKED, -URBANICITY)

# Data formatting
train_process2 <- train_process1 %>%
  mutate(AGE_IMPUTE = ifelse(is.na(AGE),
                             mean(train_process1$AGE, na.rm = TRUE),
                             AGE),
         YOJ_IMPUTE = ifelse(is.na(YOJ),
                             mean(train_process1$YOJ, na.rm = TRUE),
                             YOJ),
         CAR_AGE_IMPUTE = ifelse(is.na(CAR_AGE),
                                 mean(train_process1$CAR_AGE, na.rm = TRUE),
                                 CAR_AGE),
         HOME_VAL_IMPUTE = ifelse(is.na(HOME_VAL_NUM),
                                  mean(train_process1$HOME_VAL_NUM,
                                       na.rm = TRUE),
                                  HOME_VAL_NUM),
         INCOME_IMPUTE = ifelse(is.na(INCOME_NUM),
                                mean(train_process1$INCOME_NUM, na.rm = TRUE),
                                INCOME_NUM),
         CAR_AGE_FLR = ifelse(CAR_AGE_IMPUTE < 1, 1, CAR_AGE_IMPUTE),
         CAR_AGE_ONE_IND = ifelse(CAR_AGE_FLR == 1, 1, 0),
         HOME_VAL_ZERO_IND = ifelse(HOME_VAL_IMPUTE == 0, 1, 0),
         INCOME_ZERO_IND = ifelse(INCOME_IMPUTE == 0, 1, 0),
         YOJ_CAP = ifelse(YOJ_IMPUTE > 15, 15, YOJ_IMPUTE),
         TRAVTIME_CAP = ifelse(TRAVTIME > 65, 65, TRAVTIME),
         MVR_PTS_CAP = ifelse(MVR_PTS > 7, 7, MVR_PTS),
         INCOME_CAP = ifelse(INCOME_IMPUTE > quantile(INCOME_NUM, 0.98,
                                                      na.rm = TRUE),
                             quantile(INCOME_NUM, 0.98, na.rm = TRUE),
                             INCOME_IMPUTE),
         HOME_VAL_CAP = ifelse(HOME_VAL_IMPUTE > quantile(HOME_VAL_NUM, 0.98,
                                                          na.rm = TRUE),
                               quantile(HOME_VAL_NUM, 0.98, na.rm = TRUE),
                               HOME_VAL_IMPUTE),
         BLUEBOOK_CLOG = log(ifelse(BLUEBOOK_NUM > quantile(BLUEBOOK_NUM, 0.98,
                                                            na.rm = TRUE),
                                    quantile(BLUEBOOK_NUM, 0.98, na.rm = TRUE),
                                    BLUEBOOK_NUM)),
         OLDCLAIM_CLOG = log1p(ifelse(OLDCLAIM_NUM > quantile(OLDCLAIM_NUM,
                                                              0.98,
                                                              na.rm = TRUE),
                                      quantile(OLDCLAIM_NUM, 0.98,
                                               na.rm = TRUE),
                                      OLDCLAIM_NUM)),
         AGE_BIN = ntile(AGE_IMPUTE, 10),
         KIDSDRIV_IND = ifelse(KIDSDRIV > 1, 1, KIDSDRIV),
         CLM_FREQ_IND = ifelse(CLM_FREQ > 1, 1, CLM_FREQ),
         CAR_TYPE_GRP = ifelse(as.character(CAR_TYPE) == "Panel Truck",
                               "PT_Van",
                               ifelse(as.character(CAR_TYPE) == "Van",
                                      "PT_Van",
                                      as.character(CAR_TYPE)))) %>%
  select(-AGE, -YOJ, -CAR_AGE, -HOME_VAL_NUM, -INCOME_NUM, -CAR_AGE_IMPUTE,
         -TRAVTIME, -INCOME_IMPUTE, -HOME_VAL_IMPUTE, -BLUEBOOK_NUM,
         -KIDSDRIV, -HOMEKIDS, -CLM_FREQ, -CAR_TYPE, -MVR_PTS, -YOJ_IMPUTE,
         -OLDCLAIM_NUM)

test_process2 <- test_process1 %>%
  mutate(AGE_IMPUTE = ifelse(is.na(AGE),
                             mean(train_process1$AGE, na.rm = TRUE),
                             AGE),
         YOJ_IMPUTE = ifelse(is.na(YOJ),
                             mean(train_process1$YOJ, na.rm = TRUE),
                             YOJ),
         CAR_AGE_IMPUTE = ifelse(is.na(CAR_AGE),
                                 mean(train_process1$CAR_AGE, na.rm = TRUE),
                                 CAR_AGE),
         HOME_VAL_IMPUTE = ifelse(is.na(HOME_VAL_NUM),
                                  mean(train_process1$HOME_VAL_NUM,
                                       na.rm = TRUE),
                                  HOME_VAL_NUM),
         INCOME_IMPUTE = ifelse(is.na(INCOME_NUM),
                                mean(train_process1$INCOME_NUM, na.rm = TRUE),
                                INCOME_NUM),
         CAR_AGE_FLR = ifelse(CAR_AGE_IMPUTE < 1, 1, CAR_AGE_IMPUTE),
         CAR_AGE_ONE_IND = ifelse(CAR_AGE_FLR == 1, 1, 0),
         HOME_VAL_ZERO_IND = ifelse(HOME_VAL_IMPUTE == 0, 1, 0),
         INCOME_ZERO_IND = ifelse(INCOME_IMPUTE == 0, 1, 0),
         YOJ_CAP = ifelse(YOJ_IMPUTE > 15, 15, YOJ_IMPUTE),
         TRAVTIME_CAP = ifelse(TRAVTIME > 65, 65, TRAVTIME),
         MVR_PTS_CAP = ifelse(MVR_PTS > 7, 7, MVR_PTS),
         INCOME_CAP = ifelse(INCOME_IMPUTE > quantile(INCOME_NUM, 0.98,
                                                      na.rm = TRUE),
                             quantile(INCOME_NUM, 0.98, na.rm = TRUE),
                             INCOME_IMPUTE),
         HOME_VAL_CAP = ifelse(HOME_VAL_IMPUTE > quantile(HOME_VAL_NUM, 0.98,
                                                          na.rm = TRUE),
                               quantile(HOME_VAL_NUM, 0.98, na.rm = TRUE),
                               HOME_VAL_IMPUTE),
         BLUEBOOK_CLOG = log(ifelse(BLUEBOOK_NUM > quantile(BLUEBOOK_NUM, 0.98,
                                                            na.rm = TRUE),
                                    quantile(BLUEBOOK_NUM, 0.98, na.rm = TRUE),
                                    BLUEBOOK_NUM)),
         OLDCLAIM_CLOG = log1p(ifelse(OLDCLAIM_NUM > quantile(OLDCLAIM_NUM,
                                                              0.98,
                                                              na.rm = TRUE),
                                      quantile(OLDCLAIM_NUM, 0.98,
                                               na.rm = TRUE),
                                      OLDCLAIM_NUM)),
         AGE_BIN = ntile(AGE_IMPUTE, 10),
         KIDSDRIV_IND = ifelse(KIDSDRIV > 1, 1, KIDSDRIV),
         CLM_FREQ_IND = ifelse(CLM_FREQ > 1, 1, CLM_FREQ),
         CAR_TYPE_GRP = ifelse(as.character(CAR_TYPE) == "Panel Truck",
                               "PT_Van",
                               ifelse(as.character(CAR_TYPE) == "Van",
                                      "PT_Van",
                                      as.character(CAR_TYPE)))) %>%
  select(-AGE, -YOJ, -CAR_AGE, -HOME_VAL_NUM, -INCOME_NUM, -CAR_AGE_IMPUTE,
         -TRAVTIME, -INCOME_IMPUTE, -HOME_VAL_IMPUTE, -BLUEBOOK_NUM,
         -KIDSDRIV, -HOMEKIDS, -CLM_FREQ, -CAR_TYPE, -MVR_PTS, -YOJ_IMPUTE,
         -OLDCLAIM_NUM)

# Likelihood encoding
train_age_bins <- train_process2 %>%
  select(AGE_IMPUTE, TARGET_FLAG) %>%
  mutate(AGE_BIN = ntile(AGE_IMPUTE, 10)) %>%
  group_by(AGE_BIN) %>%
  summarise(AGE_LE = mean(TARGET_FLAG))

train_education <- train_process2 %>%
  select(EDUCATION, TARGET_FLAG) %>%
  group_by(EDUCATION) %>%
  summarise(EDUCATION_LE = mean(TARGET_FLAG))

train_job <- train_process2 %>%
  select(JOB, TARGET_FLAG) %>%
  group_by(JOB) %>%
  summarise(JOB_LE = mean(TARGET_FLAG))

train_car_type <- train_process2 %>%
  select(CAR_TYPE_GRP, TARGET_FLAG) %>%
  group_by(CAR_TYPE_GRP) %>%
  summarise(CAR_TYPE_LE = mean(TARGET_FLAG))

test_process3 <- test_process2 %>%
  left_join(., train_age_bins, by = c("AGE_BIN")) %>%
  left_join(., train_education, by = c("EDUCATION")) %>%
  left_join(., train_job, by = c("JOB")) %>%
  left_join(., train_car_type, by = c("CAR_TYPE_GRP")) %>%
  select(-AGE_BIN)

scores <- test_process3 %>%
  mutate(P_TARGET_FLAG_1 = -2.483003597310 -
           0.054573707554 * TIF +
           0.156175626835 * PARENT1_IND -
           0.625290547649 * MARRIED_IND +
           0.613124974465 * COMM_USE_IND +
           0.730838376558 * REVOKED_IND -
           2.346282191463 * RURAL_IND +
           0.523619559214 * INCOME_ZERO_IND +
           0.017839396142 * YOJ_CAP +
           0.016774384176 * TRAVTIME_CAP +
           0.098285854937 * MVR_PTS_CAP -
           0.000002665866 * INCOME_CAP -
           0.000001017080 * HOME_VAL_CAP -
           0.268592168258 * BLUEBOOK_CLOG +
           0.728686098030 * KIDSDRIV_IND +
           0.420053793383 * CLM_FREQ_IND +
           3.386556243405 * AGE_LE +
           2.668357569001 * EDUCATION_LE +
           2.614630061699 * JOB_LE +
           4.717621474153 * CAR_TYPE_LE,
         P_TARGET_FLAG = exp(P_TARGET_FLAG_1) / (1 + exp(P_TARGET_FLAG_1))) %>%
  mutate(Masters_IND = ifelse(EDUCATION == "Masters", 1, 0),
         PhD_IND = ifelse(EDUCATION == "PhD", 1, 0),
         Doctor_IND = ifelse(JOB == "Doctor", 1, 0),
         Manager_IND = ifelse(JOB == "Manager", 1, 0),
         Sport_IND = ifelse(CAR_TYPE_GRP == "Sports Car", 1, 0),
         SUV_IND = ifelse(CAR_TYPE_GRP == "z_SUV", 1, 0),
         P_TARGET_AMT = exp(6.26917773 -
                              0.12756159 * MARRIED_IND +
                              0.18093361 * MALE_IND -
                              0.11888771 * REVOKED_IND -
                              0.02637962 * CAR_AGE_FLR -
                              0.17258351 * CAR_AGE_ONE_IND -
                              0.09826503 * HOME_VAL_ZERO_IND +
                              0.02140739 * MVR_PTS_CAP +
                              0.26788307 * BLUEBOOK_CLOG +
                              0.15227087 * Masters_IND +
                              0.32114827 * PhD_IND -
                              0.39891518 * Doctor_IND -
                              0.19331633 * Manager_IND +
                              0.16454011 * Sport_IND +
                              0.11345302 * SUV_IND)) %>%
  select(INDEX, P_TARGET_FLAG, P_TARGET_AMT)

# Save output
write.csv(scores, "Kessler_test_scores.csv", row.names = FALSE)
