# Housing-Market-Forecast


## Dataset: House Prices - Advanced Regression Techniques

This is used for predicting house prices using advanced regression techniques. It contains four files:

1. **train.csv**: The training set used to build the predictive model.
2. **test.csv**: The test set used to evaluate the model's performance.
3. **data_description.txt**: A detailed description of each column in the dataset, originally prepared by Dean De Cock and slightly edited to match the column names used.
4. **sample_submission.csv**: A benchmark submission file from a linear regression on year and month of sale, lot square footage, and number of bedrooms.

### Dataset Link
Public dataset on housing prices: [house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

**Files**: 4 files  
**Size**: 957.39 kB  
**Type**: csv, txt  
**License**: MIT

This dataset is essential for anyone looking to apply advanced regression techniques in predicting house prices and includes comprehensive details about each feature used in the analysis.


### Introduction

The `Housing-Market-Forecast` appears to be a project aimed at predicting housing market prices using machine learning models. The project involves the following key components:

1. **Project Overview**:
   - The project aims to forecast housing market prices using machine learning models.

2. **Main Components**:
   - **`deploy.py`**: This is a Streamlit application for forecasting housing prices. It loads a pre-trained model, takes user input for housing features, and predicts the price.
   - **`train.ipynb`**: This notebook is used for training the machine learning models. It includes data preprocessing, model training, and evaluation steps. Two models, Linear Regression and Random Forest, are trained, and the Random Forest model is saved as `model.joblib`.
   - **`test.csv`**: This file contains test data used to validate the model's predictions. It includes various features related to housing properties.

3. **Workflow**:
   - **Data Preprocessing**: Handling missing values and encoding categorical variables.
   - **Model Training**: Using `train.ipynb`, models are trained on the dataset, and the best-performing model (Random Forest) is saved.
   - **Prediction**: The `deploy.py` script uses the saved model to make predictions based on user inputs through a Streamlit web interface.

4. **Languages Used**:
   - The repository primarily uses Jupyter Notebook (75.9%) and Python (24.1%).
  
This project provides a comprehensive solution for predicting housing prices by integrating data preprocessing, model training, and a user-friendly web application for making predictions.


### Relevance

The `Housing-Market-Forecast` project is relevant for several reasons:

1. **Practical Application**: It provides a practical solution for predicting housing market prices, which can be beneficial for real estate agents, buyers, and sellers.
2. **Machine Learning Implementation**: The project demonstrates the use of machine learning techniques, including data preprocessing, model training, and evaluation, which can be educational for those learning about data science and machine learning.
3. **Interactive Tool**: By using Streamlit for deployment, the project offers an interactive web application, making it user-friendly and accessible to non-technical users.
4. **Open Source Contribution**: Being hosted on GitHub, it allows other developers to contribute, improve, or use the project as a reference for similar tasks.

Overall, the project combines data science, machine learning, and web application development to address a real-world problem, making it a valuable resource in both academic and professional contexts.


### Deploy

The `deploy.py` is a Streamlit application designed to forecast housing market prices. Here is a detailed explanation of the file:

1. **Imports**:
   - `streamlit as st`: Used to create the web application.
   - `pandas as pd`: For data manipulation and analysis.
   - `joblib`: To load the pre-trained machine learning model.
   - `locale`: For setting locale-specific formatting.
   - `LabelEncoder` from `sklearn.preprocessing`: For encoding categorical variables.

2. **Streamlit Page Configuration**:
   - The page title is set to "Housing Market Forecast".
   - The page icon is set to a house emoji.
   - The layout is set to wide.
   - The sidebar state is set to expanded.

3. **Model and Data Loading**:
   - The pre-trained model is loaded using `joblib.load()` from a file named `model.joblib`.
   - A dataset named `test.csv` is loaded to extract feature names and encode categorical variables.

4. **Feature Extraction and Encoding**:
   - Feature names are extracted from the dataset, excluding the `Id` column.
   - Categorical variables are encoded using `LabelEncoder`. Each categorical column is transformed, and the encoders are stored in a dictionary for later use.

5. **Custom CSS Styling**:
   - Custom CSS is added to style the app and set a background image.
   - Buttons and input labels are also styled.

6. **Streamlit App Layout**:
   - The app title is set to "Housing Market Forecast".
   - Input fields are created for each feature in the dataset. For categorical features, a dropdown (select box) is provided with options. For numerical features, a number input field is provided with the average value as the default.

7. **Prediction**:
   - Input data from the user is collected and converted into a DataFrame.
   - Categorical features are encoded using the previously stored encoders.
   - When the "Predict" button is clicked, the model makes a prediction based on the input data.
   - The predicted price in USD is converted to INR using a specified conversion rate (82.50 in this case).
   - Locale-specific formatting is applied to display the predicted values in USD and INR.

8. **Output**:
   - The estimated cost is displayed in both USD and INR with appropriate formatting.



### Model

The `model.joblib` file is a serialized machine learning model saved using the `joblib` library in Python. This file is used in the `deploy.py` script to load the pre-trained model and make predictions. Here is a detailed explanation of what this file typically contains and how it is used:

1. **Model Training**:
   - During the development phase, a machine learning model is trained on a dataset using a specific algorithm (e.g., linear regression, decision tree, random forest).
   - The model learns patterns and relationships within the data to make predictions.

2. **Serialization with Joblib**:
   - Once the model is trained, it is serialized (saved) using the `joblib` library. This process converts the model into a format that can be stored on disk and later loaded back into memory.
   - Example code to save a model:
     ```python
     from sklearn.ensemble import RandomForestRegressor
     import joblib

     model = RandomForestRegressor()
     model.fit(X_train, y_train)
     joblib.dump(model, 'model.joblib')
     ```

3. **Loading the Model**:
   - In the `deploy.py` script, the model is loaded using `joblib.load()`. This deserialization process restores the model to its original state so it can be used to make predictions.
   - Example code to load the model:
     ```python
     import joblib

     model = joblib.load('model.joblib')
     ```

4. **Making Predictions**:
   - The loaded model is used to make predictions on new data. The `deploy.py` script collects input data from the user, processes it, and passes it to the model for prediction.
   - Example code to make predictions:
     ```python
     prediction = model.predict(new_data)
     ```

5. **Model Contents**:
   - The `model.joblib` file contains the learned parameters of the machine learning model, which include coefficients, weights, and any other model-specific information.
   - It also includes any pre-processing steps that were applied to the data, such as scaling or encoding.


### Dataset Contents

The `test.csv` contains data likely used for testing or validating the housing market forecasting model. This dataset includes various features related to housing properties. Here is a detailed explanation of the file:

1. **Columns (Features)**:
   - The first row of the CSV file lists the column names, which represent the different features of the housing properties. Some of the key features include:
     - `Id`: Unique identifier for each property.
     - `MSSubClass`: Identifies the type of dwelling involved in the sale.
     - `MSZoning`: Identifies the general zoning classification of the sale.
     - `LotFrontage`: Linear feet of street connected to the property.
     - `LotArea`: Lot size in square feet.
     - `Street`: Type of road access to the property.
     - `Alley`: Type of alley access to the property.
     - `LotShape`: General shape of property.
     - `LandContour`: Flatness of the property.
     - `Utilities`: Type of utilities available.
     - `LotConfig`: Lot configuration.
     - `LandSlope`: Slope of property.
     - `Neighborhood`: Physical locations within Ames city limits.
     - `Condition1` and `Condition2`: Proximity to various conditions (e.g., railroad, main road).
     - `BldgType`: Type of dwelling.
     - `HouseStyle`: Style of dwelling.
     - `OverallQual`: Rates the overall material and finish of the house.
     - `OverallCond`: Rates the overall condition of the house.
     - `YearBuilt`: Original construction date.
     - `YearRemodAdd`: Remodel date.
     - `RoofStyle`: Type of roof.
     - `RoofMatl`: Roof material.
     - `Exterior1st` and `Exterior2nd`: Exterior covering on house.
     - `MasVnrType`: Masonry veneer type.
     - `MasVnrArea`: Masonry veneer area in square feet.
     - `ExterQual`: Evaluates the quality of the material on the exterior.
     - `ExterCond`: Evaluates the present condition of the material on the exterior.
     - `Foundation`: Type of foundation.
     - `BsmtQual`: Height of the basement.
     - `BsmtCond`: General condition of the basement.
     - `BsmtExposure`: Walkout or garden level basement walls.
     - `BsmtFinType1` and `BsmtFinType2`: Quality of basement finished area.
     - `BsmtFinSF1` and `BsmtFinSF2`: Type 1 and 2 finished square feet.
     - `BsmtUnfSF`: Unfinished square feet of basement area.
     - `TotalBsmtSF`: Total square feet of basement area.
     - `Heating`: Type of heating.
     - `HeatingQC`: Heating quality and condition.
     - `CentralAir`: Central air conditioning.
     - `Electrical`: Electrical system.
     - `1stFlrSF`: First floor square feet.
     - `2ndFlrSF`: Second floor square feet.
     - `LowQualFinSF`: Low quality finished square feet (all floors).
     - `GrLivArea`: Above grade (ground) living area square feet.
     - `BsmtFullBath` and `BsmtHalfBath`: Basement full and half bathrooms.
     - `FullBath` and `HalfBath`: Above grade full and half bathrooms.
     - `Bedroom`: Number of bedrooms above basement level.
     - `Kitchen`: Number of kitchens.
     - `KitchenQual`: Kitchen quality.
     - `TotRmsAbvGrd`: Total rooms above grade (excludes bathrooms).
     - `Functional`: Home functionality rating.
     - `Fireplaces`: Number of fireplaces.
     - `FireplaceQu`: Fireplace quality.
     - `GarageType`: Garage location.
     - `GarageYrBlt`: Year garage was built.
     - `GarageFinish`: Interior finish of the garage.
     - `GarageCars`: Size of garage in car capacity.
     - `GarageArea`: Size of garage in square feet.
     - `GarageQual`: Garage quality.
     - `GarageCond`: Garage condition.
     - `PavedDrive`: Paved driveway.
     - `WoodDeckSF`: Wood deck area in square feet.
     - `OpenPorchSF`: Open porch area in square feet.
     - `EnclosedPorch`: Enclosed porch area in square feet.
     - `3SsnPorch`: Three-season porch area in square feet.
     - `ScreenPorch`: Screen porch area in square feet.
     - `PoolArea`: Pool area in square feet.
     - `PoolQC`: Pool quality.
     - `Fence`: Fence quality.
     - `MiscFeature`: Miscellaneous feature not covered in other categories.
     - `MiscVal`: Value of miscellaneous feature.
     - `MoSold`: Month sold.
     - `YrSold`: Year sold.
     - `SaleType`: Type of sale.
     - `SaleCondition`: Condition of sale.

2. **Data Entries**:
   - Each subsequent row in the CSV file represents a different property, with values corresponding to the features listed in the first row.
   - Example data entries include various combinations of zoning types, lot areas, building types, and other features.

3. **Use in Model**:
   - This dataset is likely used to test or validate the model's predictions. The model uses these features to predict housing prices or other related outcomes.
   - The features cover a wide range of property characteristics, which helps the model to make accurate predictions.


### Model Training

The `train.ipynb` is used for training a machine learning model to predict housing prices. Here is a detailed explanation:

1. **Markdown Cell**:
   - Explains that the code has been corrected with redundant imports removed and the dataset loaded only once.

2. **Code Cells**:
   - **Imports**:
     - Imports necessary libraries such as `pandas`, `joblib`, and various modules from `sklearn` for data preprocessing, model training, and evaluation.
   - **Loading the Dataset**:
     - Loads the dataset from a local file (`train.csv`).
     - Extracts and prints the feature names.
     - Handles missing values by filling them with the mode (for categorical data) or mean (for numerical data).
     - Encodes categorical variables using `LabelEncoder`.
   - **Data Splitting and Preprocessing**:
     - Splits the data into features (`X`) and target (`y`), excluding the `Id` and `SalePrice` columns from `X`.
     - Identifies categorical and numerical features.
     - Creates a `ColumnTransformer` to handle preprocessing (imputing missing values and encoding categorical variables).
     - Splits the data into training and testing sets.
   - **Model Training and Evaluation**:
     - Defines two regression models: `LinearRegression` and `RandomForestRegressor`.
     - Creates a pipeline for each model, combining the preprocessor and the model.
     - Fits the pipeline on the training data and makes predictions on the test data.
     - Calculates and prints evaluation metrics (MAE, MSE, RMSE, R-squared) for each model.
     - Saves the `RandomForest` model and preprocessor using `joblib` if it performs well.
   - **Making Predictions on Test Data**:
     - Loads the test dataset and handles missing values.
     - Encodes categorical variables using the previously fitted `LabelEncoder`.
     - Drops the `Id` column from the test data.
     - Loads the saved model and makes predictions on the test dataset.
     - Prints the predicted values for the test dataset.
