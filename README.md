# Car Price Prediction

This project aims to predict car prices based on various features using machine learning techniques. The dataset used for this project is `CarPrice_Assignment.csv`.

## Dataset Description

The dataset consists of 205 observations and 26 features. Some of the features include:

- `CarName`: The name of the car.
- `fueltype`: The type of fuel used by the car (e.g., gas, diesel).
- `aspiration`: The aspiration type of the car (e.g., std, turbo).
- `doornumber`: Number of doors in the car.
- `carbody`: The body style of the car.
- `drivewheel`: Type of drive wheel (e.g., front-wheel-drive, rear-wheel-drive).
- `enginelocation`: Location of the car engine (front or rear).
- `wheelbase`, `carlength`, `carwidth`, `carheight`: Various dimensions of the car.
- `curbweight`: The weight of the car without occupants or baggage.
- `enginetype`: Type of engine.
- `cylindernumber`: Number of cylinders in the engine.
- `enginesize`: Size of the car engine.
- `fuelsystem`: Type of fuel system used in the car.
- `boreratio`, `stroke`, `compressionratio`, `horsepower`, `peakrpm`, `citympg`, `highwaympg`: Various technical specifications of the car.
- `price`: The target variable, i.e., the price of the car.

## Data Preprocessing

### Handling Duplicate Rows and Unnecessary Columns

Duplicate rows are dropped from the dataset using `drop_duplicates` function. Additionally, unnecessary columns such as `car_ID` and `symboling` are dropped using `drop(columns=...)` method.

### Encoding Categorical Variables and Scaling Numerical Variables

Categorical variables are encoded using `LabelEncoder` to convert them into numerical format. Numerical variables are scaled using `StandardScaler` to bring them into a similar range.

## Model Selection (RandomForestRegressor)

The RandomForestRegressor algorithm is chosen for this regression task. It is a powerful ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and prevent overfitting.

## Model Evaluation

The model is evaluated using Mean Squared Error (MSE) and R-squared (R2) metrics. MSE measures the average squared difference between the actual and predicted values, while R2 indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Results

- **Accuracy**: The accuracy of the model on the test set is approximately 95.55%.
- **Mean Squared Error (MSE)**: The MSE on the test set is approximately 0.055.
- **R-squared (R2) Score**: The R2 score on the test set is approximately 0.956.

## Saving the Model

The trained model is saved using the `joblib.dump()` function to a file named `car_price_prediction_model.pkl` for future use.

## Dependencies

- pandas
- scikit-learn
- joblib

## Usage

1. Clone the repository:

git clone https://github.com/mannan-python-developer/Car-Price-Prediction-Model.git
cd car-price-prediction

2. Run the notebook or script to train and evaluate the model.

3. To use the trained model for predictions:

```bash
import joblib

# Load the model
model = joblib.load('car_price_prediction_model.pkl')

# Make predictions
predictions = model.predict(new_data)
```

## Contributors

Abdul Mannan

This README.md file provides a comprehensive overview of your project, including dataset description, data preprocessing steps, model selection, evaluation metrics, results, usage instructions, dependencies, and contributors.
