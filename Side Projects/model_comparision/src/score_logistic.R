# Create the capability to run on any provided dataframe.
# Scaling and thresholds for business decisions are out of scope.
# Input: a dataframe matching the default data dictionary.
# Output: a raw score of the likelihood to default.

library(dplyr)

# Conformance Functions

# Conform SEX to binary - 0 (female), 1 (male)
flag_sex <- function(x) {
  return(ifelse(x == 2, 0, 1))
}

# Group and conform EDUCATION
# GraduateSchoolOther (0, 1, 4, 5, 6), University (2), HighSchoolOther (3)
group_education <- function(x) {
  return(ifelse(x == 2, "University",
                ifelse(x == 3, "HighSchool", "GraduateSchoolOther")))
}

# Group and conform MARRIAGE - Married (1), Single (2), Other (0, 3)
group_marriage <- function(x) {
  return(ifelse(x == 1, "Married",
                ifelse(x == 2, "Single", "Other")))
}

# Conform data based on data quality check
conform <- function(data) {
  # Group and conform categorical variables
  # Drop partition variables
  # Rename PAY_0 to PAY_1 for consistency
  conformed_data <- data %>%
    mutate(SEX = flag_sex(SEX),
           EDUCATION = group_education(EDUCATION),
           MARRIAGE = group_marriage(MARRIAGE)) %>%
    select(-u, -train, -test, -validate, -data.group) %>%
    rename("PAY_1" = PAY_0,
           "SEX_MALE" = SEX,
           "EDUCATION_GRP" = EDUCATION,
           "MARRIAGE_GRP" = MARRIAGE)
  return(conformed_data)
}

# Feature Definitions

# Average Payment Amount
avg_pmt_amt_f  <- function(data) {
  data %>% mutate(Avg_Pmt_Amt = rowMeans(cbind(PAY_AMT1, PAY_AMT2, 
                                               PAY_AMT3, PAY_AMT4, 
                                               PAY_AMT5, PAY_AMT6))) %>%
    select(Avg_Pmt_Amt)
}

# Marriage group other with single
marriage_ind_f  <- function(data) {
  data %>% mutate(marriage_ind = ifelse(MARRIAGE_GRP == "Married", 1, 0)) %>%
    select(marriage_ind)
}

# SEX_MALE and Age interaction indicator for ages older than 34 
sex_age_int_ind_f <- function(data) {
  data %>% mutate(sex_age_int_ind = ifelse(SEX_MALE == 1 & AGE > 34, 1, 0)) %>%
    select(sex_age_int_ind)
}

# Age binned based on woeBinning tree method and WOE coded
age_bin_tree_c_f  <- function(data) {
  data %>% mutate(age_bin_tree_c = case_when(AGE <= 24 ~ -28.595798,
                                             AGE == 25 ~ -11.100857,
                                             AGE > 25 & AGE <= 34 ~ 15.236291,
                                             AGE > 34 & AGE <= 48 ~ -1.767913,
                                             AGE > 48 ~ -17.032273)) %>%
    select(age_bin_tree_c)
}

# Average Utilization using manual binnings based on cuts and WOE coded
avg_util_bin_c_f  <- function(data) {
  data %>% mutate(Util_1 = BILL_AMT1 / LIMIT_BAL,
                  Util_2 = BILL_AMT2 / LIMIT_BAL,
                  Util_3 = BILL_AMT3 / LIMIT_BAL,
                  Util_4 = BILL_AMT4 / LIMIT_BAL,
                  Util_5 = BILL_AMT5 / LIMIT_BAL,
                  Util_6 = BILL_AMT6 / LIMIT_BAL,
                  Avg_Util = rowMeans(cbind(Util_1, Util_2, Util_3, 
                                            Util_4, Util_5, Util_6)),
                  prelim = case_when(Avg_Util <= 0 ~ 0,
                                     Avg_Util > 1 ~ 1,
                                     TRUE ~ Avg_Util),
                  avg_util_bin_c = case_when(prelim <= 0.0009562517806 ~ -46.875314,
                                             prelim <= 0.009194422492 ~ 3.158137,
                                             prelim <= 0.03113875 ~ 40.423949,
                                             prelim <= 0.200677265 ~ 59.245509,
                                             prelim <= 0.3673418095 ~ 29.244568,
                                             prelim <= 0.5317903333 ~ -3.252604,
                                             prelim <= 0.8231566667 ~ -20.440143,
                                             prelim <= 0.9589748333 ~ -49.898714,
                                             prelim > 0.9589748333 ~ -62.270139)) %>%
    select(avg_util_bin_c)
}

# Utilization Growth Over 6 Months using woeBinning with zero distinct and manual adjustment then coded
util_growth_6mo_woe_c_f  <- function(data) {
  data %>% mutate(Util_1 = BILL_AMT1 / LIMIT_BAL,
                  Util_6 = BILL_AMT6 / LIMIT_BAL,
                  Util_Growth_6mo = Util_1 - Util_6,
                  util_growth_6mo_woe_c = case_when(Util_Growth_6mo == 0 ~ -56.45760,
                                                    Util_Growth_6mo <= -0.02930625 ~ -36.58428,
                                                    Util_Growth_6mo <= 0.72262 ~ 26.57977,
                                                    Util_Growth_6mo > 0.72262 ~ -30.56202)) %>%
    select(util_growth_6mo_woe_c)
}

# Max Bill Amount using woeBinning and coded
max_bill_amt_woe_c_f  <- function(data) {
  data %>% mutate(Max_Bill_Amt = pmax(BILL_AMT1, BILL_AMT2, BILL_AMT3, 
                                      BILL_AMT4, BILL_AMT5, BILL_AMT6),
                  max_bill_amt_woe_c = case_when(Max_Bill_Amt <= 550 ~ -56.818324,
                                                 Max_Bill_Amt <= 4015.85 ~ -22.323103,
                                                 Max_Bill_Amt <= 6986 ~ 26.131410,
                                                 Max_Bill_Amt <= 52552.8 ~ -3.704997,
                                                 Max_Bill_Amt > 52552.8 ~ 18.395324)) %>%
    select(max_bill_amt_woe_c)
}

# Max Delinquency capped at two
max_dlq_cap_woe_f  <- function(data) {
  data %>% mutate(PAY_1_FLR = ifelse(PAY_1 < 0, 0, PAY_1), 
                  PAY_2_FLR = ifelse(PAY_2 < 0, 0, PAY_2),
                  PAY_3_FLR = ifelse(PAY_3 < 0, 0, PAY_3),
                  PAY_4_FLR = ifelse(PAY_4 < 0, 0, PAY_4),
                  PAY_5_FLR = ifelse(PAY_5 < 0, 0, PAY_5),
                  PAY_6_FLR = ifelse(PAY_6 < 0, 0, PAY_6),
                  prelim = pmax(PAY_1_FLR, PAY_2_FLR, PAY_3_FLR, 
                                PAY_4_FLR, PAY_5_FLR, PAY_6_FLR),
                  max_dlq_cap = ifelse(prelim > 3, 3, prelim),
                  max_dlq_cap_woe = case_when(max_dlq_cap == 0 ~ 73.74673,
                                              max_dlq_cap == 1 ~ -17.55100,
                                              max_dlq_cap == 2 ~ -97.36615,
                                              max_dlq_cap == 3 ~ -184.307870)) %>%
    select(max_dlq_cap_woe)
}

# PAY_1 capped at 3 WOE coded
pay_1_cap_woe_f <- function(data) {
  data %>% 
    mutate(pay_1_cap = ifelse(PAY_1 > 3, 3, PAY_1),
           pay_1_cap_woe = case_when(pay_1_cap == -2 ~ 63.74593,
                                     pay_1_cap == -1 ~ 35.01887,
                                     pay_1_cap == 0 ~ 61.76402,
                                     pay_1_cap == 1 ~ -54.87779,
                                     pay_1_cap == 2 ~ -207.76782,
                                     pay_1_cap == 3 ~ -235.6497)) %>%
    select(pay_1_cap_woe)
}

# Indicator for whether max_dlq occurred after the first month
max_dlq_cap_id_f  <- function(data) {
  data %>% mutate(PAY_1_FLR = ifelse(PAY_1 < 0, 0, PAY_1), 
                  PAY_2_FLR = ifelse(PAY_2 < 0, 0, PAY_2),
                  PAY_3_FLR = ifelse(PAY_3 < 0, 0, PAY_3),
                  PAY_4_FLR = ifelse(PAY_4 < 0, 0, PAY_4),
                  PAY_5_FLR = ifelse(PAY_5 < 0, 0, PAY_5),
                  PAY_6_FLR = ifelse(PAY_6 < 0, 0, PAY_6),
                  prelim = pmax(PAY_1_FLR, PAY_2_FLR, PAY_3_FLR, 
                                PAY_4_FLR, PAY_5_FLR, PAY_6_FLR),
                  max_dlq_cap_id = ifelse(prelim == PAY_6_FLR, 0, 1)) %>%
    select(max_dlq_cap_id)
}

# Apply feature functions for candidate features
apply_features <- function(data, y_values = FALSE) {
  # If y-values is TRUE, actual DEFAULT is on the data set.
  # This allows scoring to be used for monitoring.
  if (y_values == TRUE) {
    data %>%
      do(cbind(.[, which(names(.) %in% c("ID", "DEFAULT", "SEX_MALE", 
                                         "LIMIT_BAL"))],
               age_bin_tree_c_f(.),
               sex_age_int_ind_f(.),
               marriage_ind_f(.),
               avg_pmt_amt_f(.),
               avg_util_bin_c_f(.),
               util_growth_6mo_woe_c_f(.),
               max_bill_amt_woe_c_f(.),
               max_dlq_cap_woe_f(.),
               pay_1_cap_woe_f(.),
               max_dlq_cap_id_f(.))) %>%
      select(-DEFAULT, DEFAULT)
  } else {
    data %>%
      do(cbind(.[, which(names(.) %in% c("ID", "SEX_MALE", "LIMIT_BAL"))],
               age_bin_tree_c_f(.),
               sex_age_int_ind_f(.),
               marriage_ind_f(.),
               avg_pmt_amt_f(.),
               avg_util_bin_c_f(.),
               util_growth_6mo_woe_c_f(.),
               max_bill_amt_woe_c_f(.),
               max_dlq_cap_woe_f(.),
               pay_1_cap_woe_f(.),
               max_dlq_cap_id_f(.)))
  }
}


# Model Scorecard returning IDs and predicted values
score_glm <- function(data, y_values = FALSE) {
  # If y-values is TRUE, actual DEFAULT is on the data set.
  # This allows scoring to be used for monitoring.
  prelim_data <- data %>%
    mutate(coef_product = -0.9895088502382 +
             -0.0000009720075 * LIMIT_BAL +
              0.0234022999342 * SEX_MALE +
             -0.0011872300666 * age_bin_tree_c +
              0.1382745370286 * marriage_ind +
             -0.0000153046048 * Avg_Pmt_Amt +
             -0.0040687927169 * avg_util_bin_c +
             -0.0030657636751 * util_growth_6mo_woe_c +
             -0.0071983126793 * max_bill_amt_woe_c +
             -0.0060045372548 * max_dlq_cap_woe +
             -0.0066660425588 * pay_1_cap_woe +
             -0.4609398062202 * max_dlq_cap_id +
              0.2309030887678 * sex_age_int_ind,
           raw_score = exp(coef_product) / (exp(coef_product) + 1))
  
  if (y_values == TRUE) {
    prelim_data %>%
      select(ID, raw_score, DEFAULT)
  } else {
    prelim_data %>%
      select(ID, raw_score)
  }
}

# Example when scoring validation data
if (sys.nframe() == 0) {
  # Time process  
  start_time <- Sys.time()
  
  credit_card_default <- readRDS("./data/raw/credit_card_default.RData")
  
  validation_data <- credit_card_default %>%
    filter(data.group == 3)
  
  post_conform <- conform(validation_data)
  post_feature <- apply_features(post_conform, y_values = TRUE)
  post_score <- score_glm(post_feature, y_values = TRUE)
  
  end_time <- Sys.time()
  
  print(paste0("Scoring Time (seconds): ", round(end_time - start_time, 2)))
}
