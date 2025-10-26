if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(ggcorrplot)

dl <- tempfile()
download.file("https://github.com/jeffreyrchin/Spotify-Dataset/raw/master/archive.zip", dl)

Spotify <- read.csv(unzip(dl, "data.csv"))

# Test set will be 10% of Spotify data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = Spotify$popularity, times = 1, p = 0.1, list = FALSE)
train <- Spotify[-test_index,]
temp <- Spotify[test_index,]

# Make sure energy and year entries in test set are also in the train set
test <- temp %>% 
  semi_join(train, by = "energy") %>%
  semi_join(train, by = "year")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

rm(dl, temp, test_index, removed)

# Data Exploration

# Pre-processing. Removing unused variables:

Spotify <- Spotify %>% select(-c(artists, id, name, release_date))

# Structure:

dim(Spotify)

str(Spotify)

# Data Visualizations:

# Number of Observations for Each Popularity Ranking:

Spotify %>% count(popularity) %>% 
  ggplot(aes(popularity, n)) + 
  geom_col(color = "black") + 
  scale_x_continuous(breaks = seq(0, 100, 5)) +
  ggtitle("Observation Count for Each Popularity Ranking")

# Number of observations for high popularity rankings (above 80):

Spotify %>% count(popularity) %>% tail(20)

# Popularity rankings grouped by energy level, totaled:

Spotify %>% group_by(energy) %>% 
  summarize(n = n()) %>%
  ggplot(aes(energy, n)) + 
  xlab("Energy") +
  geom_point() +
  ggtitle("Total observations by energy level")

# Popularity rankings grouped by energy level, averaged:

Spotify %>% group_by(energy) %>% 
  summarize(avg = mean(popularity)) %>% 
  ggplot(aes(energy, avg)) + 
  xlab("energy") +
  geom_point() + 
  ggtitle("Popularity grouped by energy, averaged")

# Popularity rankings grouped by year, totaled:

Spotify %>% group_by(year) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(year, n)) + 
  xlab("year") +
  geom_line() + 
  ggtitle("Total observations grouped by year")

# Popularity rankings grouped by year, averaged:

Spotify %>% group_by(year) %>% 
  summarize(avg = mean(popularity)) %>% 
  ggplot(aes(year, avg)) + 
  xlab("year") +
  geom_point() + 
  ggtitle("Popularity grouped by year, averaged")

# Correlation table:

cor(Spotify) %>% ggcorrplot(lab = T, lab_size = 2, outline.color = "black")

# Loss Function (RMSE):

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Model 1: Just The Average

mu_hat <- mean(train$popularity) # Average of all popularity ratings of tracks in train

prediction_1 <- mu_hat # using just the average to predict popularity ratings in test
prediction_1

model_1_rmse <- RMSE(test$popularity, prediction_1) # Calculate the RMSE
model_1_rmse

# RMSE of using any other number besides the average popularity rating:

RMSE(test$popularity, 40)
RMSE(test$popularity, 40) > model_1_rmse
RMSE(test$popularity, 1)
RMSE(test$popularity, 1) > model_1_rmse

# Model 2: Energy Effect Model

b_j_hat <- train %>%
  group_by(energy) %>% 
  summarize(b_j_hat = mean(popularity - mu_hat)) # b_j_hat represents the adjusted average popularity rating for each energy level centered around 0 (b_j of 0 implies a popularity rating of mu_hat)

b_j_hat %>% 
  ggplot(aes(b_j_hat)) + 
  geom_histogram(bins = 4, color = "black") +
  ggtitle("Histogram of b_j_hat")

b_j_hat_test <- test %>% 
  left_join(b_j_hat, by = "energy") %>%
  .$b_j_hat # b_j estimates for each energy level in test

prediction_2 <- mu_hat + b_j_hat_test # combining the average and the average popularity rating bias to predict popularity ratings in test

model_2_rmse <- RMSE(test$popularity, prediction_2) # Calculate the RMSE
model_2_rmse

# Model 3: Energy and Year Effect Model

b_y_hat <- train %>% 
  left_join(b_j_hat, by = "energy") %>%
  group_by(year) %>% 
  summarize(b_y_hat = mean(popularity - mu_hat - b_j_hat)) # b_y_hat represents the adjusted average popularity rating for each track release year centered around 0.

b_y_hat %>% 
  ggplot(aes(b_y_hat)) + 
  geom_histogram(bins = 5, color = "black") +
  ggtitle("Histogram of b_y_hat")

b_y_hat_test <- test %>% 
  left_join(b_y_hat, by = "year") %>%
  .$b_y_hat # b_y estimates for each track release year in test

prediction_3 <- mu_hat + b_j_hat_test + b_y_hat_test

model_3_rmse <- RMSE(test$popularity, prediction_3) # Calculate the RMSE
model_3_rmse

# Model 4: Regularized Energy and Year Effect Model

Regularize <- function(lambda, train_set, test_set) { # Function used to choose the best lambda given a train and test set.
  
  b_j_hat_train <- train_set %>% 
    group_by(energy) %>% 
    summarize(b_j_hat = sum(popularity - mu_hat)/(n() + lambda))
  
  b_y_hat_train <- train_set %>% 
    left_join(b_j_hat, by = "energy") %>%
    group_by(year) %>% 
    summarize(b_y_hat = sum(popularity - mu_hat - b_j_hat)/(n() + lambda))
  
  prediction <- test_set %>% 
    left_join(b_j_hat_train, by = "energy") %>%
    left_join(b_y_hat_train, by = "year") %>%
    mutate(pred = mu_hat + b_j_hat + b_y_hat) %>%
    pull(pred)
  
  RMSE(test_set$popularity, prediction)
  
}

# Using cross-validation to choose the best lambda:
rmses <- sapply(seq(0, 10, 0.25), Regularize, train, test)

Lambda <- seq(0, 10, 0.25)[which.min(rmses)] # lambda that yields the lowest RMSE
Lambda

plot(rmses, x = seq(0, 10, 0.25), xlab = "lambdas")

model_4_rmse <- min(rmses) # RMSE calculated using best lambda
model_4_rmse

# Model 5: Random Forest Model
# takes a few minutes to run
set.seed(1, sample.kind = "Rounding") # if using R 3.5 or earlier, use set.seed(1)
control <- trainControl(method = "cv", number = 5, p = 0.8)
Train_fit <- train(popularity ~ energy + year, 
                   method = "rf", 
                   ntree = 2, 
                   data = train, 
                   trControl = control, 
                   tuneGrid = data.frame(mtry = seq(1, 2, 1)))

model_5_rmse <- RMSE(test$popularity, predict(Train_fit, test))
model_5_rmse

RMSE_table <- data.table(methods = c("Using mu_hat", 
                                     "Movie effect model", 
                                     "Movie + user effect model", 
                                     "Movie + user + year effect model", 
                                     "Regularized movie + user + year effect model"), 
                         RMSEs = c(model_1_rmse, 
                                   model_2_rmse, 
                                   model_3_rmse, 
                                   model_4_rmse, 
                                   model_5_rmse), 
                         "Improvement Percentage" = c(NA, 
                                                      (1 - (model_2_rmse / model_1_rmse)) * 100,
                                                      (1 - (model_3_rmse / model_2_rmse)) * 100,
                                                      (1 - (model_4_rmse / model_3_rmse)) * 100,
                                                      (1 - (model_5_rmse / model_4_rmse)) * 100))

knitr::kable(RMSE_table)
