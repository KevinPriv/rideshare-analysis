# loading the data
train_data<-read.csv("RideShare_train.csv")
test_data<-read.csv("RideShare_test.csv")

# combining data sets
train_data$isTraining<-TRUE
test_data$isTraining<-FALSE

combined_data<- rbind(train_data, test_data)

#dimensions of data
dim(combined_data)
#summary of data
summary(combined_data)
#datatypes within the data
sapply(combined_data, class)
# looking at the data
head(combined_data, 15)
# correcting datatypes after being loaded in by csv
combined_data$cab_type <- as.factor(combined_data$cab_type)
combined_data$name <- as.factor(combined_data$name)
combined_data$short_summary <- as.factor(combined_data$short_summary)
# looking at corrected data
str(combined_data)
# looking at levels
levels(combined_data$cab_type)
levels(combined_data$name)
levels(combined_data$short_summary)
# --- MISSINGNESS
library("mice")
#md.pattern(combined_data, rotate.names=TRUE)
percentage_missing <-  colMeans(is.na(combined_data[,c("timestamp", "datetime", "price")])) * 100
percentage_missing

# removing missing data
combined_data = combined_data[,-2] # remove timestamp
combined_data = combined_data[,-5] # remove datetime
combined_data = na.omit(combined_data) # removes any rows containing null values (price)
#md.pattern(combined_data, rotate.names=TRUE)

# summarize the class distribution
percentage_cab_type <- prop.table(table(combined_data$cab_type)) * 100
cbind(freq=table(combined_data$cab_type), percentage=percentage_cab_type)

percentage_cab_name <- prop.table(table(combined_data$name)) * 100
cbind(freq=table(combined_data$name), percentage=percentage_cab_name)

percentage_summary <- prop.table(table(combined_data$short_summary)) * 100
cbind(freq=table(combined_data$short_summary), percentage=percentage_summary)

#------------EDA

# visualizing data
library(ggplot2)

ggplot(combined_data,aes(name,price))+geom_boxplot() +
  labs(title = "Boxplot of each taxi type in respect to price")
  
ggplot(combined_data,aes(factor(surge_multiplier),price))+geom_boxplot() +
  labs(title = "Boxplot of surge multiplier in respect to price")

ggplot(combined_data, aes(short_summary)) + geom_bar() +
  labs(title = "Bar graph showing weather vs amount of rides", x ="Weather", y ="Rides")

ggplot(combined_data, aes(factor(day))) + geom_bar() +
  labs(title = "Amount of rides per day", x = "day of month", y = "rides")


# correlation 
numeric_cols <- sapply(combined_data, is.numeric)
data_numeric = combined_data[, numeric_cols]
summary(data_numeric)
data_corr = cor(data_numeric)

library(corrplot)
corrplot(data_corr)
corrplot(data_corr, method="number") 

#removing multicollinearity
combined_data = combined_data[, !names(combined_data) %in% c("apparentTemperature", "temperatureHigh", "temperatureLow", "dewPoint", "temperatureMin","temperatureMax", "windGust", "visibility.1", "day", "month")]
head(combined_data)

#----SCALING
library(dplyr)
#scale numeric values
combined_data %>% 
  mutate_if(is.numeric, scale)
head(combined_data, 15)

#change ID to rownames
rownames(combined_data) <- combined_data$id
combined_data$id <- NULL
head(combined_data)

#removing outliers
numeric_data = select_if(combined_data, is.numeric)
model = lm(formula = price~., data = numeric_data)
summary(model)
cooks_dist <- cooks.distance(model)
influence <- cooks_dist[(cooks_dist > (10 * mean(cooks_dist)))] # removes if 10 times bigger than mean cooks distance
influential_rides <- names(influence)

cleaned_combined_data = combined_data[!rownames(combined_data) %in% influential_rides,]
nrow(combined_data)
nrow(cleaned_combined_data)
head(cleaned_combined_data)
nrow(cleaned_combined_data[cleaned_combined_data$isTraining==FALSE, ])
# PART 2

library(randomForest)
library(caret)

stratified_df <- cleaned_combined_data %>%
  group_by(source, destination, cab_type, name, surge_multiplier) %>% #any number of variables you wish to partition by proportionally
  sample_frac(0.3)# 30% of the dataset

train<-stratified_df[stratified_df$isTraining==TRUE,]
test<-stratified_df[stratified_df$isTraining==FALSE, ]
#remove variable isTrain from both train and test
train$isTraining<-NULL
test$isTraining<-NULL

summary(train)
set.seed(123)

# cross validation
cv_control <- trainControl(method = "cv", number = 10, verboseIter = TRUE) # 10 fold cross validation
summary(train)
nrow(train)

#----LINEAR
linear_model <- train(price ~ ., data = train, method = "lm", trControl = cv_control)
summary(linear_model)
linear_predictions <- predict(linear_model, newdata = test)
linear_residuals <- test$price - linear_predictions
linear_mse <- mean((linear_predictions - test$price)^2)
linear_rmse <- sqrt(linear_mse)
linear_mae <- mean(abs(linear_predictions - test$price))
linear_rsquared = postResample(pred = linear_predictions, obs = test$price)

# histogram of residuals
linear_res_hist <- ggplot() + geom_histogram(aes(linear_residuals), bins = 60) + labs(title = "Histogram of residuals (LR)")

# RESIDUAL PLOT
linear_res_plot <- ggplot() +
  geom_point(aes(x = linear_predictions, y = linear_residuals)) +
  geom_hline(yintercept = 0, linetype = "dashed", color="blue") +
  labs(x = "Predicted Values", y = "Residuals", title = "Residual Plot (LR)")

# predicted vs original plot
linear_plot <- ggplot() +
  geom_point(aes(x = test$price, y = linear_predictions)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color="blue", size=1.5) +
  labs(x = "Actual Price", y = "Predictions Price", title = "Actual vs Prediction plot (LR)")


#-------DECISION TREE
dt_grid <- expand.grid(maxdepth=c(7, 10, 12))
dt_model <- train(price ~ ., data = train, method = "rpart2", trControl = cv_control, tuneGrid=dt_grid)
summary(dt_model)
dt_importance <- varImp(dt_model)
plot(dt_model)
plot(dt_importance)
#library(rpart.plot)
#rpart.plot(dt_model$finalModel)
dt_predictions <- predict(dt_model, newdata = test)
dt_residuals <- test$price - dt_predictions
dt_mse <- mean((dt_predictions - test$price)^2)
dt_rmse <- sqrt(dt_mse)
dt_mae <- mean(abs(dt_predictions - test$price))
dt_rsquared = postResample(pred = dt_predictions, obs = test$price)

# histogram of residuals
dt_res_hist <- ggplot() + geom_histogram(aes(dt_residuals), bins = 60) + labs(title = "Histogram of residuals (DT)")

# RESIDUAL PLOT
dt_res_plot <- ggplot() +
  geom_point(aes(x = dt_predictions, y = dt_residuals)) +
  geom_hline(yintercept = 0, linetype = "dashed", color="blue") +
  labs(x = "Predicted Values", y = "Residuals", title = "Residual Plot (DT)")

# predicted vs original plot
dt_plot <- ggplot() +
  geom_point(aes(x = test$price, y = dt_predictions)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color="blue", size=1.5) +
  labs(x = "Actual Price", y = "Predictions Price", title = "Actual vs Prediction plot (DT)")


#--------RF
rf_grid <- expand.grid(mtry=c(4, 6, 8))
rf_model <- train(price ~ ., data = train, method = "rf", trControl = cv_control, tuneGrid=rf_grid)
summary(rf_model)
importance <- varImp(rf_model)
plot(rf_model)
plot(importance)
rf_predictions <- predict(rf_model, newdata = test)
rf_residuals <- test$price - rf_predictions
rf_mse <- mean((rf_predictions - test$price)^2)
rf_rmse <- sqrt(rf_mse)
rf_mae <- mean(abs(rf_predictions - test$price))
rf_rsquared = postResample(pred = rf_predictions, obs = test$price)

# histogram of residuals
rf_res_hist <- ggplot() + geom_histogram(aes(rf_residuals), bins = 60) + labs(title = "Histogram of residuals (RF)")

# RESIDUAL PLOT
rf_res_plot <- ggplot() +
  geom_point(aes(x = rf_predictions, y = rf_residuals)) +
  geom_hline(yintercept = 0, linetype = "dashed", color="blue") +
  labs(x = "Predicted Values", y = "Residuals", title = "Residual Plot (RF)")

# predicted vs original plot
rf_plot <-ggplot() +
  geom_point(aes(x = test$price, y = rf_predictions)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color="blue", size=1.5) +
  labs(x = "Actual Price", y = "Predictions Price", title = "Actual vs Prediction plot (RF)")

#---------------KNN
knn_grid <- expand.grid(k=c(10, 50, 100, 200))
knn_model <- train(price ~ ., data = train, method = "knn", trControl = cv_control, tuneGrid=knn_grid)
summary(knn_model)
plot(knn_model)
knn_predictions <- predict(knn_model, newdata = test)
knn_residuals <- test$price - knn_predictions
knn_mse <- mean((knn_predictions - test$price)^2)
knn_rmse <- sqrt(knn_mse)
knn_mae <- mean(abs(knn_predictions - test$price))
knn_rsquared = postResample(pred = knn_predictions, obs = test$price)

# histogram of residuals
knn_res_hist <- ggplot() + geom_histogram(aes(knn_residuals), bins = 60) + labs(title = "Histogram of residuals (KNN)")

# RESIDUAL PLOT
knn_res_plot <- ggplot() +
  geom_point(aes(x = knn_predictions, y = knn_residuals)) +
  geom_hline(yintercept = 0, linetype = "dashed", color="blue") +
  labs(x = "Predicted Values", y = "Residuals", title = "Residual Plot (KNN)")

# predicted vs original plot
knn_plot <- ggplot() +
  geom_point(aes(x = test$price, y = knn_predictions)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color="blue", size=1.5) +
  labs(x = "Actual Price", y = "Predictions Price", title = "Actual vs Prediction plot (KNN)")

#----------NEURALNET
nn_grid <- expand.grid(size = c(5,10,25,50), decay=c(0, 0.1)) 
nn_model <- train(price ~ ., data = train, method = "nnet", trControl = cv_control, tuneGrid = nn_grid, MaxNWts = 5000, maxit = 500)

plot(nn_model)

nn_predictions <- predict(nn_model, newdata = test)
nn_residuals <- test$price - nn_predictions
nn_mse <- mean((nn_predictions - test$price)^2)
nn_rmse <- sqrt(nn_mse)
nn_mae <- mean(abs(nn_predictions - test$price))
nn_rsquared = postResample(pred = nn_predictions, obs = test$price)

# histogram of residuals
nn_res_hist <- ggplot() + geom_histogram(aes(nn_residuals), bins = 60) + labs(title = "Histogram of residuals (NN)")

# RESIDUAL PLOT
nn_res_plot <- ggplot() +
  geom_point(aes(x = nn_predictions, y = nn_residuals)) +
  geom_hline(yintercept = 0, linetype = "dashed", color="blue") +
  labs(x = "Predicted Values", y = "Residuals", title = "Residual Plot (NN)")

# predicted vs original plot
nn_plot <- ggplot() +
  geom_point(aes(x = test$price, y = nn_predictions)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color="blue", size=1.5) +
  labs(x = "Actual Price", y = "Predictions Price", title = "Actual vs Prediction plot (NN)")


#--- GRIDS
library(gridExtra)
grid.arrange(linear_plot, dt_plot, rf_plot, knn_plot, nn_plot, ncol=2)
grid.arrange(linear_res_plot, dt_res_plot, rf_res_plot, knn_res_plot, nn_res_plot, ncol=2)



