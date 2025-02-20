# NBA-Player-Props
This project focuses on predicting the future performance of NBA players based on historical game data. The model utilizes XGBoost to forecast specific player statistics (such as points, assists, rebounds, etc.) for upcoming games.

**Project Overview**
The project consists of two primary Python scripts:
**fetch_historical_data.py:**
  Scrapes historical NBA player boxscore data from the NBA's official website.
  Retrieves data for multiple seasons, which is then saved into a CSV file for use in the model.
**xgboost_model.py:**
  Uses the scraped data to train an XGBoost model to predict a player's performance in upcoming games.
  It performs hyperparameter tuning and cross-validation to ensure the best model performance.
  Includes a feature that allows the user to predict whether a player's stat will go over or under a specific sportsbook line.

**Current State**
While the model is functional and offers a basic approach to predicting player statistics, it is still in the early stages and has room for improvement. The following areas could be enhanced:
Feature Engineering: More features, such as recent performance trends, opponent strength, and other contextual factors, could be integrated.
Model Tuning: The model could benefit from further hyperparameter optimization and trying different machine learning algorithms.
Data Quality: Currently, the data is scraped and cleaned in a basic manner. Handling edge cases and improving data preprocessing could lead to more accurate predictions.

