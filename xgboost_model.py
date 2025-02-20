import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

def preprocess_data(data):
    columns_to_keep = ['Points', '3PM', 'REB', 'AST', 'TO', 'STL', 'BLK']
    data = data[columns_to_keep]
    
    # Replace non-numeric values with NaN
    data = data.replace({'-': pd.NA, '': pd.NA, None: pd.NA})
    
    # Convert columns to numeric values, coercing errors to NaN
    for col in ['3PM', 'REB', 'AST', 'TO', 'STL', 'BLK', 'Points']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Fill NaN values with 0
    data = data.fillna(0)
    
    return data

def train_model(player_name, stat, file_path='nba_player_boxscores_multiple_seasons.csv'):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset file at {file_path} was not found.")

    data = pd.read_csv(file_path)

    # Filter data for the specific player
    player_data = data[data['Player'] == player_name]
    
    # Check if there is enough data for the player
    if player_data.empty:
        raise ValueError(f"No data found for player: {player_name}")

    # Preprocess the data
    player_data = preprocess_data(player_data)

    # Define features and target
    features = ['3PM', 'REB', 'AST', 'TO', 'STL', 'BLK']
    X = player_data[features]
    y = player_data[stat]

    # Initialize XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror')

    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'max_depth': [5, 10, 50],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2, n_jobs=-1)
    random_search.fit(X, y)

    # Get the best model
    best_model = random_search.best_estimator_

    # Evaluate model using cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='neg_mean_squared_error')
    print(f'Cross-validation scores (negative MSE): {cv_scores}')
    print(f'Mean cross-validation score (negative MSE): {cv_scores.mean()}')
    print(f'Standard deviation of cross-validation scores: {cv_scores.std()}')

    # Convert negative MSE to positive for interpretation
    cv_scores = -cv_scores

    # Additional model evaluation metrics
    predictions = best_model.predict(X)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared (R2): {r2}')

    # Save the trained model to a file
    pickle.dump(best_model, open(f'model_{player_name}_{stat}_xgboost.pkl', 'wb'))

    return best_model

def predict_over_under(player_name, stat, line, file_path='nba_player_boxscores_multiple_seasons.csv'):
    # Check if model exists
    model_file = f'model_{player_name}_{stat}_xgboost.pkl'
    if not os.path.exists(model_file):
        print(f"Model for {player_name} and {stat} not found. Training new model...")
        model = train_model(player_name, stat, file_path)
    else:
        model = pickle.load(open(model_file, 'rb'))

    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset file at {file_path} was not found.")
    
    data = pd.read_csv(file_path)
    player_data = data[data['Player'] == player_name].tail(1)

    
    if player_data.empty:
        raise ValueError(f"No data found for player: {player_name}")

    
    player_data = preprocess_data(player_data)

    
    features = ['3PM', 'REB', 'AST', 'TO', 'STL', 'BLK']
    X = player_data[features]

    
    prediction = model.predict(X)

    print(f"Predicted {stat} for {player_name}: {prediction[0]:.2f}")

    # Make the over/under recommendation
    if prediction > line:
        print(f"Recommendation: Bet the over ({prediction[0]:.2f} > {line})")
    else:
        print(f"Recommendation: Bet the under ({prediction[0]:.2f} < {line})")

if __name__ == '__main__':
    player_name = input("Enter player name: ")
    stat = input("Enter performance metric (Points, TO, BLK, STL, AST, REB, 3PM): ")
    line = float(input(f"Enter the sportsbook line for {stat}: "))
    
    print("Model loaded successfully.")
    predict_over_under(player_name, stat, line)