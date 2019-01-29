library(tidyverse)
library(caret)
library(e1071)
library(gridExtra)
library(scales)
# CYO Project
data = read.csv('Admission_Predict.csv')
# Looking at the data
colnames(data)
# Checking the histogram of each of the columns
data %>% select(-Serial.No.) %>% gather() %>% ggplot(aes(value)) + facet_wrap(~ key, scales = 'free') +
  geom_histogram()
# This looks nice, we can see all the columns are numeric, but not all of 
# them are continuous, so we need to take care when evaluating them.
# Let's create a correlation matrix to see the weights of our analysis. 
# Every colunm is import for this analysis, except serial.no., so it should
# be removed.
cor(data %>% select(-Serial.No.))
# Great, but we can see that even though each column is important, the
# continuous columns don't have the same range. To make this work we should
# change this. The CGPA, GRE.Score and TOEFL.Score will range from 1 to 100.
new_data <- data %>%
  mutate(CGPA = rescale(CGPA, to = c(0, 100)),
         GRE.Score = rescale(GRE.Score, to = c(0,100)),
         TOEFL.Score = rescale(TOEFL.Score, to = c(0, 100))) %>%
  mutate(CGPA = CGPA - mean(CGPA), GRE.Score = GRE.Score - mean(GRE.Score),
         TOEFL.Score = TOEFL.Score - mean(TOEFL.Score),
         SOP = factor(SOP), University.Rating = factor(University.Rating),
         Research = factor(Research), LOR = factor(LOR))
# Now let's separate the data into test_set and train_set
test_index <- createDataPartition(data$Chance.of.Admit, times = 1, p = 0.2, list = F)
test_set <- new_data[test_index,]
train_set <- new_data[-test_index,]
# what if I change the chance of admit to a factor, values bigger than 0.8
# are approved(1) and lower than 0.8 are not approved(0)
factor_train_set <- train_set %>%
  mutate(Chance.of.Admit = factor(ifelse(Chance.of.Admit < 0.8, 0, 1)))
factor_test_set <- test_set %>%
  mutate(Chance.of.Admit = factor(ifelse(Chance.of.Admit < 0.8, 0, 1)))
# Glm
glm_train <- train(Chance.of.Admit ~. , method = 'glm', data = factor_train_set)
glm_pred <- predict(glm_train, factor_test_set)
glm_confusion <- confusionMatrix(glm_pred, factor_test_set$Chance.of.Admit)
glm_plot <- data.frame(glm_confusion$table) %>% ggplot(aes(Reference, Prediction)) +
  geom_tile(aes(fill = -Freq)) +
  geom_text(aes(Reference, Prediction, label = Freq)) + 
  ggtitle('GLM Results')

# Using this factors set, we will train Naive Bayes
nb_train <- train(Chance.of.Admit ~. , method = 'nb', data = factor_train_set)
nb_pred <- predict(nb_train, factor_test_set)
nb_confusion <- confusionMatrix(nb_pred, factor_test_set$Chance.of.Admit)
nb_plot <- data.frame(nb_confusion$table) %>% ggplot(aes(Reference, Prediction)) +
  geom_tile(aes(fill = -Freq)) +
  geom_text(aes(Reference, Prediction, label = Freq)) + 
  ggtitle('NB Results')

# There is a big improvement from the NB method over the glm method
# Now for KNN
knn_train <- train(Chance.of.Admit ~., method = 'knn', data = factor_train_set,
                   tuneGrid = data.frame(k = seq(5, 51, 2)))
knn_pred <- predict(knn_train, factor_test_set)
knn_confusion <- confusionMatrix(knn_pred, factor_test_set$Chance.of.Admit)
# plot showing the tuning parameters
plot(knn_train)
knn_plot <- data.frame(knn_confusion$table) %>% ggplot(aes(Reference, Prediction)) +
  geom_tile(aes(fill = -Freq)) +
  geom_text(aes(Reference, Prediction, label = Freq)) + 
  ggtitle('KNN Results')

# Random forest
rf_train <- train(Chance.of.Admit ~., method = 'rf', data = factor_train_set)
rf_pred <- predict(rf_train, factor_test_set)
rf_confusion <- confusionMatrix(rf_pred, factor_test_set$Chance.of.Admit)
rf_plot <- data.frame(rf_confusion$table) %>% ggplot(aes(Reference, Prediction)) +
  geom_tile(aes(fill = -Freq)) +
  geom_text(aes(Reference, Prediction, label = Freq)) + 
  ggtitle('RF Results')

# Creating a ensemble
ensemble_df <- data.frame(rf_pred, knn_pred, nb_pred, glm_pred)
votes <- rowMeans(ensemble_df == '1')
ensemble_pred <- ifelse(votes > 0.5, '1', '0')
ensemble_confusion <- confusionMatrix(factor(ensemble_pred), factor_test_set$Chance.of.Admit)
ensemble_plot <- data.frame(ensemble_confusion$table) %>% ggplot(aes(Reference, Prediction)) +
  geom_tile(aes(fill = -Freq)) +
  geom_text(aes(Reference, Prediction, label = Freq)) + 
  ggtitle('Ensemble Results')


# Looking at all confusion matrix plots at once.
grid.arrange(rf_plot, nb_plot, rf_plot, ensemble_plot, glm_plot, knn_plot)

# Creating the table
enb_df <- data.frame(naive_bayes = nb_confusion$byClass[1:3], ensemble = ensemble_confusion$byClass[1:3],
                     knn = knn_confusion$byClass[1:3], glm = glm_confusion$byClass[1:3],
                     random_forest = rf_confusion$byClass[1:3])
enb_df['Accuracy',] <- c(nb_confusion$overall['Accuracy'], ensemble_confusion$overall['Accuracy'],
                         knn_confusion$overall['Accuracy'], glm_confusion$overall['Accuracy'],
                         rf_confusion$overall['Accuracy'])
enb_df
# Thanks to Mohan S Acharya, Asfia Armaan, Aneeta S Antony :
# A Comparison of Regression Models for Prediction of Graduate Admissions,
# IEEE International Conference on Computational Intelligence in Data Science 2019
# for providing the dataset