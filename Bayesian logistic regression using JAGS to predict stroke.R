dat <- read.csv("healthcare-dataset-stroke-data.csv", 
                na.strings = "N/A")
dat <- na.omit(dat)
dat <- subset(dat, select = -id)
head(dat)
categorical_vars <- c('gender', 'hypertension', 'heart_disease', 'ever_married',
                      'work_type', 'Residence_type', 'smoking_status', 'stroke')

dat[categorical_vars] <- lapply(dat[categorical_vars], as.factor)
head(dat)

library(GGally)


dat$gender <- as.numeric(as.factor(dat$gender))
dat$hypertension <- as.numeric(as.factor(dat$hypertension))
dat$heart_disease <- as.numeric(as.factor(dat$heart_disease))
dat$ever_married <- as.numeric(as.factor(dat$ever_married))
dat$work_type <- as.numeric(as.factor(dat$work_type))
dat$Residence_type <- as.numeric(as.factor(dat$Residence_type))
dat$smoking_status <- as.numeric(as.factor(dat$smoking_status))


n_levels_gender <- length(levels(factor(dat$gender)))
n_levels_hypertension <- length(levels(factor(dat$hypertension)))
n_levels_heart_disease <- length(levels(factor(dat$heart_disease)))
n_levels_ever_married <- length(levels(factor(dat$ever_married)))
n_levels_work_type <- length(levels(factor(dat$work_type)))
n_levels_Residence_type <- length(levels(factor(dat$Residence_type)))
n_levels_smoking_status <- length(levels(factor(dat$smoking_status)))

# Standardize
dat$age <- scale(dat$age)
dat$avg_glucose_level <- scale(dat$avg_glucose_level)
dat$bmi <- scale(dat$bmi)

library(caret)

# Stratified sampling
set.seed(123)
train_index <- createDataPartition(dat$stroke, p = 0.7, list = FALSE)
train_data <- dat[train_index, ]
test_data <- dat[-train_index, ]

table(train_data$stroke)
table(test_data$stroke)
str(train_data)

library(smotefamily)

# SMOTE
smote_data <- SMOTE(train_data[ , -which(names(train_data) == "stroke")], train_data$stroke)
balanced_train_data <- smote_data$data

# Check sample distribution
table(balanced_train_data$class)
balanced_train_data$stroke <- balanced_train_data$class
balanced_train_data$class <- NULL  # 删除多余的 class 变量
head(balanced_train_data)

# Convert categorical variable 
balanced_train_data$stroke <- as.integer(balanced_train_data$stroke)
balanced_train_data$gender <- as.integer(balanced_train_data$gender)
balanced_train_data$hypertension <- as.integer(balanced_train_data$hypertension)
balanced_train_data$heart_disease <- as.integer(balanced_train_data$heart_disease)
balanced_train_data$ever_married <- as.integer(balanced_train_data$ever_married)
balanced_train_data$work_type <- as.integer(balanced_train_data$work_type)
balanced_train_data$Residence_type <- as.integer(balanced_train_data$Residence_type)
balanced_train_data$smoking_status <- as.integer(balanced_train_data$smoking_status)

train_data$stroke <- as.integer(train_data$stroke)
train_data$gender <- as.integer(train_data$gender)
train_data$hypertension <- as.integer(train_data$hypertension)
train_data$heart_disease <- as.integer(train_data$heart_disease)
train_data$ever_married <- as.integer(train_data$ever_married)
train_data$work_type <- as.integer(train_data$work_type)
train_data$Residence_type <- as.integer(train_data$Residence_type)
train_data$smoking_status <- as.integer(train_data$smoking_status)

train_data$stroke <- ifelse(train_data$stroke == 2, 1, 0)
str(train_data)
table(train_data$stroke)

library(rjags)

set.seed(113)

model_string = "
model {
  for (i in 1:length(stroke)) {
    # Logistic regression model for binary classification problems

    stroke[i] ~ dbern(p[i])
    logit(p[i]) <- b0 + b_age * age[i] + b_glucose * avg_glucose_level[i] + b_bmi * bmi[i] +
                   b_gender[gender[i]] + b_hypertension[hypertension[i]] +
                   b_heart_disease[heart_disease[i]] + b_ever_married[ever_married[i]] +
                   b_work_type[work_type[i]] + b_Residence_type[Residence_type[i]] +
                   b_smoking_status[smoking_status[i]]

  }

  b0 ~ dnorm(0, 1e-6)
  b_age ~ dnorm(0, 1e-6)
  b_glucose ~ dnorm(0, 1e-6)
  b_bmi ~ dnorm(0, 1e-6)

  for (j in 1:n_levels_gender) {
    b_gender[j] ~ dnorm(0, 1e-6)
  }
  for (j in 1:n_levels_hypertension) {
    b_hypertension[j] ~ dnorm(0, 1e-6)
  }
  for (j in 1:n_levels_heart_disease) {
    b_heart_disease[j] ~ dnorm(0, 1e-6)
  }
  for (j in 1:n_levels_ever_married) {
    b_ever_married[j] ~ dnorm(0, 1e-6)
  }
  for (j in 1:n_levels_work_type) {
    b_work_type[j] ~ dnorm(0, 1e-6)
  }
  for (j in 1:n_levels_Residence_type) {
    b_Residence_type[j] ~ dnorm(0, 1e-6)
  }
  for (j in 1:n_levels_smoking_status) {
    b_smoking_status[j] ~ dnorm(0, 1e-6)
  }

  }
"

create_data_jags <- function(data) {
  list(
    N = nrow(data),
    stroke = data$stroke,
    age = data$age,
    avg_glucose_level = data$avg_glucose_level,
    bmi = data$bmi,
    gender = data$gender,
    hypertension = data$hypertension,
    heart_disease = data$heart_disease,
    ever_married = data$ever_married,
    work_type = data$work_type,
    Residence_type = data$Residence_type,
    smoking_status = data$smoking_status,
    n_levels_gender = n_levels_gender,
    n_levels_hypertension = n_levels_hypertension,
    n_levels_heart_disease = n_levels_heart_disease,
    n_levels_ever_married = n_levels_ever_married,
    n_levels_work_type = n_levels_work_type,
    n_levels_Residence_type = n_levels_Residence_type,
    n_levels_smoking_status = n_levels_smoking_status
  )
}

data_jags <- create_data_jags(balanced_train_data)

data_jags$age <- as.vector(balanced_train_data$age)
data_jags$avg_glucose_level <- as.vector(balanced_train_data$avg_glucose_level)
data_jags$bmi <- as.vector(balanced_train_data$bmi)


str(data_jags)


model <- jags.model(textConnection(model_string), data = data_jags, n.chains = 3)

# Burn-in
update(model, 3000)

# Sample
params <- c("b0","b_age", "b_glucose", "b_bmi", "b_gender", "b_hypertension", "b_heart_disease", "b_ever_married", "b_work_type", "b_Residence_type", "b_smoking_status")
samples <- coda.samples(model, variable.names = params, n.iter = 10000)

summary(samples)
plot(samples)
# Run Gelman-Rubin 
gelman.diag(samples)

# 重新采样b0, 因为之前写错了啊啊啊！！慢死了！！！
samples_b0 <- coda.samples(model, variable.names = "b0", n.iter = 10000)

b0_median <- median(as.matrix(samples_b0))
print(b0_median)


head(test_data)
str(test_data)

test_data$stroke <- as.integer(test_data$stroke)
test_data$gender <- as.integer(test_data$gender)
test_data$hypertension <- as.integer(test_data$hypertension)
test_data$heart_disease <- as.integer(test_data$heart_disease)
test_data$ever_married <- as.integer(test_data$ever_married)
test_data$work_type <- as.integer(test_data$work_type)
test_data$Residence_type <- as.integer(test_data$Residence_type)
test_data$smoking_status <- as.integer(test_data$smoking_status)


coef_median <- apply(as.matrix(samples), 2, median)
coef_median["b0"] <- b0_median


# Revise Stroke
head(test_data)
test_data$stroke <- ifelse(test_data$stroke == 2, 1, 0)

predict_function <- function(test_data, coef) {

  logit_p <- coef["b0"] + 
             coef["b_age"] * test_data$age +
             coef["b_glucose"] * test_data$avg_glucose_level +
             coef["b_bmi"] * test_data$bmi +
             coef[paste0("b_gender[", test_data$gender, "]")] +
             coef[paste0("b_hypertension[", test_data$hypertension, "]")] +
             coef[paste0("b_heart_disease[", test_data$heart_disease, "]")] +
             coef[paste0("b_ever_married[", test_data$ever_married, "]")] +
             coef[paste0("b_work_type[", test_data$work_type, "]")] +
             coef[paste0("b_Residence_type[", test_data$Residence_type, "]")] +
             coef[paste0("b_smoking_status[", test_data$smoking_status, "]")]

  p <- 1 / (1 + exp(-logit_p))

  # Apply a Threshold of 0.5 to Predict
  predicted <- ifelse(p >= 0.5, 1, 0)
  return(predicted)
}

predicted <- predict_function(test_data, coef_median)

library(caret)
confusionMatrix(factor(predicted), factor(test_data$stroke))

library(pROC)

roc_curve <- roc(test_data$stroke, predicted)  
plot(roc_curve)  
auc(roc_curve)   


data2_jags <- create_data_jags(train_data)

data2_jags$age <- as.vector(train_data$age)
data2_jags$avg_glucose_level <- as.vector(train_data$avg_glucose_level)
data2_jags$bmi <- as.vector(train_data$bmi)


str(data2_jags)

# Now, test a model on unbalanced data
model2 <- jags.model(textConnection(model_string), data = data2_jags, n.chains = 3)

update(model2, 3000)

params2 <- c("b0","b_age", "b_glucose", "b_bmi", "b_gender", "b_hypertension", "b_heart_disease", "b_ever_married", "b_work_type", "b_Residence_type", "b_smoking_status")
samples2 <- coda.samples(model2, variable.names = params2, n.iter = 10000)

summary(samples2)
plot(samples2)

gelman.diag(samples2)

coef_median2 <- apply(as.matrix(samples2), 2, median)
predicted2 <- predict_function(test_data, coef_median2)

levels(factor(predicted2))
levels(factor(test_data$stroke))

library(caret)
confusionMatrix(factor(predicted2), factor(test_data$stroke))


library(pROC)
roc_curve2 <- roc(test_data$stroke, predicted2)  
plot(roc_curve2)  
auc(roc_curve2)   


predict_function2 <- function(test_data, coef) {

  logit_p <- coef["b0"] + 
             coef["b_age"] * test_data$age +
             coef["b_glucose"] * test_data$avg_glucose_level +
             coef["b_bmi"] * test_data$bmi +
             coef[paste0("b_gender[", test_data$gender, "]")] +
             coef[paste0("b_hypertension[", test_data$hypertension, "]")] +
             coef[paste0("b_heart_disease[", test_data$heart_disease, "]")] +
             coef[paste0("b_ever_married[", test_data$ever_married, "]")] +
             coef[paste0("b_work_type[", test_data$work_type, "]")] +
             coef[paste0("b_Residence_type[", test_data$Residence_type, "]")] +
             coef[paste0("b_smoking_status[", test_data$smoking_status, "]")]


  p <- 1 / (1 + exp(-logit_p))


  return(p)
}
predicted_probabilities <- predict_function2(test_data, coef_median2)
roc_curve <- roc(test_data$stroke, predicted_probabilities)
plot(roc_curve)
auc(roc_curve)

#I'm freaking dead but balanced data perform better:)