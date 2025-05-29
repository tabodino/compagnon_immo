import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROCESSED_DATA_PATH = 'data/processed/'
MODEL_NAME = 'Voting_CatBoost_LightGBM_XGBoost'
MODEL_PATH = f'models/{MODEL_NAME}.pkl'


def display_predict_scores(y_test, y_pred):
    mae_test = mean_absolute_error(y_test, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_test = r2_score(y_test, y_pred)
    scores = pd.DataFrame([{
        'Modèle': 'Voting_CatBoost_LightGBM_XGBoost',
        'MAE test': mae_test,
        'RMSE Test': rmse_test,
        'R2 Test': r2_test
    }])
    print(scores)
    return scores


try:
    model = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(PROCESSED_DATA_PATH + 'X_test.csv')
    y_test = pd.read_csv(PROCESSED_DATA_PATH + 'y_test.csv')
    print(X_test.shape)
    print(y_test.shape)

    print('Prédictions en cours...')
    y_pred = model.predict(X_test)

    display_predict_scores(y_test, y_pred)

    X_test['y_pred'] = y_pred
    X_test.to_csv(f"{PROCESSED_DATA_PATH}X_test.csv", index=False)
    print('Prédictions sauvegardées dans X_test.csv.')

except FileNotFoundError:
    print("Le modèle n'existe pas. Veuillez d'abord l'entraîner.")
