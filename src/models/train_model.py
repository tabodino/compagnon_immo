from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt


MODELS_PATH = 'models/'
REPORTS_PATH = 'reports/figures/'
PROCESSED_DATA_PATH = 'data/processed/'


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f'Entrainement du modèle {model_name}...')
    start_time = time.time()
    # Entraînement du modèle
    model.fit(X_train, y_train.values.ravel())
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # Calcul des métriques
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # Affichage des métriques
    scores = pd.DataFrame([{
        'Modèle': model_name,
        'MAE Train': mae_train, 'MAE test': mae_test,
        'RMSE Train': rmse_train, 'RMSE Test': rmse_test,
        'R2 Train': r2_train, 'R2 Test': r2_test
    }])
    print(scores)
    display_learning_curve(model, model_name, X_train, y_train)

    check_overfitting(mae_test, mae_train)

    # Sauvegarde du modèle
    joblib.dump(model, f"{MODELS_PATH}{model_name}.pkl")
    print(f"Modèle {model_name}.pkl sauvegardé")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Temps (min) :", round(elapsed_time / 60, 2))

    return


def display_learning_curve(model, model_name, X_train, y_train):
    print(
        f"Génération de la courbe d'apprentissage pour le modèle {model_name}...")
    file = 'learning_curve_train.png'
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train,
        cv=3,
        scoring='neg_mean_absolute_error',
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=42
    )

    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(train_sizes, train_scores_mean, 'o-',
             color="r", label="Erreur MAE entraînement")
    plt.plot(train_sizes, test_scores_mean, 'o-',
             color="g", label="Erreur MAE validation")
    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Nombre d'échantillons d'entraînement")
    plt.ylabel("MAE")
    plt.legend(loc="best")
    plt.grid()

    plt.savefig(f"{REPORTS_PATH}{file}", dpi=300, bbox_inches="tight")
    print(f"Courbe d'apprentissage enregistrée sous {REPORTS_PATH}{file}")

    plt.show(block=False)
    plt.close()


def check_overfitting(mae_test, mae_train):
    print("Vérification de l'overffiting...")
    if mae_train != 0:
        overfit_ratio = mae_test / mae_train
        if overfit_ratio > 1.5:
            print(
                f"Overfitting probable détecté !"
                f"(Ratio MAE Test/Train = {overfit_ratio:.2f})")
        elif overfit_ratio < 0.7:
            print(
                f"Underfitting probable détecté !"
                f"(Ratio MAE Test/Train = {overfit_ratio:.2f})")
        else:
            print(
                f"Modèle équilibré (Ratio MAE Test/Train = {overfit_ratio:.2f})")
    else:
        print('MAE train = 0!')


catboost = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    random_state=42
)

lightgbm = LGBMRegressor(
    num_leaves=50,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1
)

xgboost = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

ensemble_model = VotingRegressor(
    estimators=[
        ('catboost', catboost),
        ('lightgbm', lightgbm),
        ('xgboost', xgboost)
    ],
    # n_jobs=-1
)

X_train = pd.read_csv(PROCESSED_DATA_PATH + 'X_train.csv')
y_train = pd.read_csv(PROCESSED_DATA_PATH + 'y_train.csv')
X_test = pd.read_csv(PROCESSED_DATA_PATH + 'X_test.csv')
y_test = pd.read_csv(PROCESSED_DATA_PATH + 'y_test.csv')

train_and_evaluate_model(
    model=ensemble_model,
    X_train=X_train,
    y_train=y_train.squeeze(),
    X_test=X_test,
    y_test=y_test.squeeze(),
    model_name="Voting_CatBoost_LightGBM_XGBoost"
)
