# **Simple Linear Regression: Experience vs. Salary**

This project demonstrates how to build and evaluate a **Simple Linear Regression** model to predict salary based on years of experience. The dataset contains information about individuals' years of experience and their corresponding salaries. We use this data to explore the relationship between experience and salary and create a predictive model.

## **Objective**
The primary goal of this project is to predict an individual's salary based on their years of experience using simple linear regression.

## **Dataset**
The dataset contains 30 records with two columns: 
- **Years of Experience**: The number of years an individual has worked.
- **Salary**: The salary of the individual in USD.

You can find the dataset on [Kaggle](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression?select=Salary_Data.csv).

## **Project Workflow**
### 1. **Import Libraries**
We start by importing the necessary libraries:
- `pandas` and `numpy` for data handling.
- `matplotlib` for visualization.
- `scikit-learn` for model building, training, and evaluation.

### 2. **Data Preprocessing**
- Load the dataset using `pandas`.
- Perform basic exploratory data analysis, like checking the shape and summary of the dataset.
- Define the feature (Years of Experience) and target (Salary).

### 3. **Data Visualization**
- Create a scatter plot to visualize the relationship between Years of Experience and Salary.
- The plot indicates a strong positive correlation between the two variables, which makes it a good candidate for simple linear regression.

### 4. **Model Building**
- Split the dataset into training and test sets (80% training, 20% testing) using `train_test_split`.
- Train the **Linear Regression** model on the training data.
- Predict the salaries for the test data using the trained model.

### 5. **Model Evaluation**
- Evaluate the performance of the model using **Mean Squared Error (MSE)** and **R-squared (R²)** metrics.
  - **Mean Squared Error**: 49,830,096.86
  - **R-squared**: 0.9024
- The model explains 90.24% of the variation in salary based on years of experience.

### 6. **Residual Analysis**
- Create a residual plot to check for random distribution of residuals.
- Randomly scattered residuals indicate that the linear regression model is a good fit.

### 7. **Visualizing Predictions**
- Create a scatter plot to compare the actual test values and the predicted values from the model.

### 8. **Equation of the Regression Line**
- The equation of the regression line is computed from the model's coefficients:
  ```
  y = 25,321.58 + 9,423.82 * x
  ```
  Where:
  - `x` is the years of experience.
  - `y` is the predicted salary.

### 9. **Regression Line vs. Original Data**
- Visualize the regression line along with the original data points to better understand the fit of the model.

## **Technologies Used**
- **Python**: Main programming language used for data processing and model building.
- **Pandas**: For data loading and manipulation.
- **Numpy**: For numerical operations.
- **Matplotlib**: For creating visualizations.
- **scikit-learn**: For building, training, and evaluating the linear regression model.

## **Results**
- The linear regression model provides a strong fit to the data with an R² value of **0.9024**. This means that 90.24% of the variance in salary is explained by years of experience.
- The regression equation can be used to predict salaries for individuals based on their years of experience.
