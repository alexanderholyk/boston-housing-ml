# train.py - baseline Linear Regression
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import fetch_openml # Fetch dataset from OpenML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def load_data():
    """Fetch Boston housing dataset from OpenML."""
    X, y = fetch_openml(name='boston', version=1, return_X_y=True, as_frame=True)
    return X.astype(np.float32), y.astype(np.float32)

def main():    
    # Load data
    X, y = load_data()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 score on test: {r2:.4f}")

    # Save the model
    Path("models").mkdir(exist_ok=True) # Ensure the models directory exists
    joblib.dump(model, "models/linear_regression.joblib") # Save the model to a file
    print("Model saved to models/linear_regression.joblib")


if __name__ == "__main__":
    main()