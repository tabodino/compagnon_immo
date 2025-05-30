import streamlit as st
import pandas as pd

IMG_FOLDER = "reports/figures/"

st.header("✨ Modélisation")

st.subheader("Modélisation du prix au m²")

st.write("#### Choix des métriques")

st.write(
    "> - **MAE :** Interprétation intuitive de l'erreur moyenne \n"
    "> - **RMSE :** Interprétation des erreurs avec écarts importants \n"
    "> - **R2 :** Interprétation de la proportion de variance expliquée\n"
    "> - **MAPE :** Comparaison de la performance entre différents modèles\n"
)

st.write("#### Modèles baseline")

st.write("Premières itérations sur les modèles suivants:")
st.write(
    "> - **LinearRegression :** Modèle simple et rapide, utile comme point de référence. \n"
    "> - **Ridge :** Ajoute une pénalisation des grands coefficients, utile pour éviter le surapprentissage en présence de multicolinéarité  \n"
    "> - **Lasso :** Pemet d'éliminer les variables non pertinentes \n"
    "> - **RandomForest :** Modèle d'ensemble robuste qui capture bien les relations complexes \n"
    "> - **DecisionTree :** Efficace pour capter les intéractions non linéaires et données hétérogènes  \n"
    "> - **XGBoost :** Modèle de gradient boosting performant et rapide, efficace sur les données avec des relations complexes"
)

data = {
    "Modèle": ["LinearRegression", "Ridge", "Lasso", "RandomForest", "XGBoost"],
    "MAE_Train": [7580.869718, 7580.849154, 7580.309918, 58.983126, 1350.271401],
    "MAE_Test": [7367.813591, 7367.792423, 7367.235078, 142.582656, 1410.821135],
    "RMSE_Train": [65371.252389, 65371.252392, 65371.253913, 4572.758073, 19885.149905],
    "RMSE_Test": [58191.826227, 58191.825446, 58191.792474, 12141.242044, 23728.173733],
    "R²_Train": [0.181351, 0.181351, 0.181351, 0.995994, 0.924250],
    "R²_Test": [0.202868, 0.202868, 0.202869, 0.965300, 0.867464],
    "Ratio_MAE_Test_Train": [0.971896, 0.971895, 0.971891, 2.417347, 1.044843],
}
df = pd.DataFrame(data)
st.dataframe(
    df.style.format(
        {"MAE": "{:,.2f}", "MSE": "{:,.2e}", "RMSE": "{:,.2f}", "R²": "{:.6f}"}
    )
)

st.write(
    "Les modèles ensemblistes semblent les plus performants lors "
    "de notre première approche naïve. Nous remarquons la tendance "
    "d'overfitting pour ces derniers. "
    "Ces scores élevés ont mis en lumière des failles dans notre "
    "méthodologie d'encodage et de standardisation, entraînant une "
    "fuite de données."
)

st.write("#### Recherche optimisations")

st.write("> **Réduction de dimension PCA**")

data_pca = {
    "Modèle": [
        "GradientBoosting",
        "XGBoost",
        "GradientBoosting",
        "XGBoost",
        "GradientBoosting",
        "XGBoost",
        "GradientBoosting",
        "XGBoost",
        "GradientBoosting",
        "XGBoost",
        "GradientBoosting",
        "XGBoost",
        "GradientBoosting",
        "XGBoost",
    ],
    "Composants": [
        "2",
        "2",
        "3",
        "3",
        "5",
        "5",
        "10",
        "10",
        "20",
        "20",
        "30",
        "30",
        "0.95%",
        "0.95%",
    ],
    "MAE Train": [
        6718.33,
        6634.21,
        5150.26,
        4952.41,
        4369.32,
        3856.17,
        4031.85,
        2943.25,
        3627.91,
        2514.93,
        3671.76,
        2418.00,
        4267.22,
        3512.43,
    ],
    "RMSE Train": [
        65502.96,
        65849.80,
        64256.77,
        63929.08,
        59007.22,
        57732.53,
        51062.28,
        37132.92,
        46576.45,
        31169.63,
        45179.59,
        28407.35,
        57880.28,
        53054.21,
    ],
    "R² Train": [
        0.1780,
        0.1693,
        0.2090,
        0.2171,
        0.3330,
        0.3615,
        0.5005,
        0.7359,
        0.5844,
        0.8139,
        0.6090,
        0.8454,
        0.3582,
        0.4608,
    ],
    "MAE Test": [
        6509.13,
        6444.81,
        4991.29,
        4819.08,
        4248.38,
        3802.20,
        4008.03,
        3258.64,
        3704.41,
        3001.13,
        3744.63,
        2938.82,
        4127.89,
        3537.62,
    ],
    "RMSE Test": [
        58424.71,
        59081.07,
        57521.17,
        57272.92,
        54346.15,
        54122.74,
        50111.91,
        47842.57,
        47716.85,
        47541.37,
        46872.95,
        46992.28,
        52265.44,
        51072.64,
    ],
    "R² Test": [
        0.1965,
        0.1783,
        0.2211,
        0.2278,
        0.3047,
        0.3104,
        0.4089,
        0.4612,
        0.4640,
        0.4680,
        0.4828,
        0.4802,
        0.3570,
        0.3860,
    ],
    "Overfit": ["non"] * 14,
}

df_pca = pd.DataFrame(data_pca)
st.dataframe(df_pca)
st.write(
    "La réduction de dimension ne semble pas affecter les performances."
    "Même sur échelle de composants allant de 2 à 30, les scores ne "
    "restent pas acceptables."
)

st.write("> **RandomSearchCV & GridSearchCV**")

st.write(
    "Le modèle GradientBossting est assez lent surtout "
    "dans un processus de recherche d'hyper-paramètres. "
    "HistGradientBoostingRegressor est plus rapide."
)


st.caption(
    "> Meilleurs paramètres pour GradientBoosting :  \n"
    "> {'max_iter': 200, 'max_depth': 7, 'learning_rate': 0.05, 'l2_regularization': 1.0}"
)

st.caption(
    "> Meilleurs paramètres pour XGBoost :  \n"
    "> {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.1}"
)

st.write("Résultats GradientBoosting")
data = {
    "Phase": ["Train", "Test"],
    "MAE": [1328.41, 1361.74],
    "RMSE": [20682.07, 23492.24],
    "R²": [0.9181, 0.8701],
}
df = pd.DataFrame(data)
st.dataframe(df)

st.write("> **Bagging et Stacking**")
st.write("Le test a été effectué sur les modèles Gradient Boosting, et XGBoost.")
data = {
    "Méthode": ["Bagging", "Bagging", "Stacking", "Stacking"],
    "Phase": ["Train", "Test", "Train", "Test"],
    "MAE": [659.18, 811.85, 869.30, 973.76],
    "RMSE": [13685.79, 22966.10, 13543.81, 20361.14],
    "R²": [0.9641, 0.8758, 0.9649, 0.9024],
}
df = pd.DataFrame(data)
st.dataframe(df)

col1, col2 = st.columns(2)
with col1:
    st.image(f"{IMG_FOLDER}learning_curve_bagging.jpg", use_container_width=True)
with col2:
    st.image(f"{IMG_FOLDER}learning_curve_stacking.jpg", use_container_width=True)

st.write(
    "Les résultats sont nettement meilleurs qu'avec une réduction de dimension."
    "Le stacking est plutôt performant."
)

st.write("#### Autres modèles")
st.write(
    "Nous allons essayer les modèles CatBoost et LightGBM. "
    "**CatBoost** est idéal pour les données avec beaucoup de "
    "variables catégorielles et offre une grande robustesse sans tuning. "
    "**LightGBM** est parfait pour les datasets volumineux, il est rapide "
    "et a une gestion efficace de la mémoire."
)

data_cat = {
    "Métrique": ["MAE", "RMSE", "R²"],
    "Valeur": [974.784179, 16712.529879, 0.934250736],
}
st.write("_CatBoost_")
st.dataframe(pd.DataFrame(data_cat))

data_gbm = {
    "Phase": ["Train", "Test"],
    "MAE": [757.43, 894.20],
    "RMSE": [11169.46, 18589.37],
    "R²": [0.9761, 0.9187],
}
st.write("_LightGBM_")
st.dataframe(pd.DataFrame(data_gbm))

col1, col2 = st.columns(2)
with col1:
    st.image(f"{IMG_FOLDER}learning_curve_catboost.jpg", use_container_width=True)
with col2:
    st.image(f"{IMG_FOLDER}learning_curve_lightgbm.jpg", use_container_width=True)

st.write("Les résultats sont plutôt bons pour ces types de modèles.")

st.write("#### Meilleures performances avec Voting Regressor")

st.code(
    """ensemble_model = VotingRegressor(
    estimators=[
        ('catboost', catboost),
        ('lightgbm', lightgbm),
        ('xgboost', xgboost)
    ],
    n_jobs=-1
) """
)

data_voting = {
    "Phase": ["Train", "Test"],
    "MAE": [871.33, 931.01],
    "RMSE": [14259.16, 17923.62],
    "R²": [0.9610, 0.9244],
}

st.dataframe(pd.DataFrame(data_voting))

st.write(
    "La combinaison de modèles permet d'avoir de "
    "belles performances. Le modèle **Voting Regressor** "
    "pourrait répondre à notre problématique."
)

st.image(f"{IMG_FOLDER}learning_curve_train.png", use_container_width=True)

st.write("#### Dernières optimisations")

st.write(
    "Nous lançons une recherche d'hyper-paramètres "
    "couplé avec un stacking regressor."
)

st.code(
    """
voting_model = VotingRegressor(
    estimators=[('catboost', catboost), ('lightgbm',
                 lightgbm), ('xgboost', xgboost)],
    weights=[0.4, 0.3, 0.3],
    n_jobs=-1
)

stacking_model = StackingRegressor(
    estimators=[('catboost', catboost), ('lightgbm',
                 lightgbm), ('xgboost', xgboost)],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1
)
 """
)

data_opt = {
    "Phase": ["Train", "Test"],
    "MAE": [1303.29, 1341.57],
    "RMSE": [17863.52, 20293.06],
    "R²": [0.9389, 0.9031],
}

st.dataframe(pd.DataFrame(data_opt))

st.write("L'ajout d'un stackingRegressor n'a pas permis d'augmenter les performances.")

st.write("#### Interprétation")

col1, col2, col3 = st.columns(3)
with col1:
    st.write("Catboost")
    st.image(f"{IMG_FOLDER}swap_catboost.jpg", use_container_width=True)
    st.image(f"{IMG_FOLDER}features_catboost.jpg", use_container_width=True)

with col2:
    st.write("LightGBM")
    st.image(f"{IMG_FOLDER}swap_lightgbm.jpg", use_container_width=True)
    st.image(f"{IMG_FOLDER}features_lightgbm.jpg", use_container_width=True)
with col3:
    st.write("XGBoost")
    st.image(f"{IMG_FOLDER}swap_xgboost.jpg", use_container_width=True)
    st.image(f"{IMG_FOLDER}features_xgboost.jpg", use_container_width=True)

st.write(
    "L'affichage de l'importance des features de shape nous "
    "montre que les variables '*lots numéros*' ne sont pas très utiles."
)

st.write("### Deep learning")

st.write(
    "Un essai a été fait avec un Perceptron multi couche "
    "pour comparer avec les résultats obtenus précédemment."
    "Les résultats sont corrects sans trop d'optimisations. "
    "La recherche d'amélioration est très coûteuse en temps et "
    "même avec plusieurs heures d'entraînement, il n'y a pas de "
    "gain de performance."
)


st.write("---")

st.subheader("Modélisation évolution des prix au m²")

st.write("#### Etude série temporelle")

st.write("Exemple pour le département 75")

st.image(f"{IMG_FOLDER}evolution_prix_dep_75.jpg", use_container_width=True)

st.write("#### Modèles baseline")

st.write(
    "Par rapport à une analyse ACF et PACF nous "
    "entrainons un modèle **SARIMA** avec les paramètres suivants :"
)

st.code(
    """
p, d, q = 1, 0, 1         # Paramètres non saisonniers
P, D, Q, s = 1, 0, 1, 12  # Paramètres saisonniers
"""
)

data = {
    "Dataset": ["Train", "Test"],
    "MAE": [109.926282, 60.208866],
    "RMSE": [153.422057, 81.795772],
    "R²": [0.190082, -0.101328],
    "MAPE (%)": [47.392081, 33.675467],
}

st.dataframe(pd.DataFrame(data))

st.image(f"{IMG_FOLDER}sarima_forecast.jpg", use_container_width=True)

st.write(
    "Les performances sont loin d'être acceptable. Après plusieurs essais "
    "SARIMA et Prophet nous avons toujours des scores R2 négatifs."
    "Nous décidons d'enrichir nos données pour améliorer la compréhension "
    "des modèles avec des données de contexte économique comme les taux "
    "bancaire, de livret A et d'inflation"
)

st.write("#### Recherche optimisations")

st.write("Nous allons essayer Prophet avec l'ajout de 'regressor'")
st.code(
    """
    model_prophet.add_regressor('taux_inflation')
    model_prophet.add_regressor('taux_livret_a')
    model_prophet.add_regressor('taux_moyen_bancaire')
"""
)

data_prophet = {
    "Métrique": ["MAE", "RMSE", "R²"],
    "Valeur": [85461.60, 104281.40, -1.23],
}

st.dataframe(pd.DataFrame(data_prophet))

st.image(f"{IMG_FOLDER}prophet_forecast.jpg", use_container_width=True)

st.write("#### Deep learning")

st.write("> GRU & LSTM")
st.write(
    "GRU utilise moins de paramètres et est plus rapide à entraîner.  \n"
    "LSTM est plus performant pour les séries temporelles complexes."
)

data = {
    "Modèle": ["GRU + LSTM"],
    "Scaler": ["Robust"],
    "MAE": [124336.189735],
    "RMSE": [185566.537182],
    "R²": [0.28966],
}

st.dataframe(pd.DataFrame(data))

st.write("> RNN")
st.code(
    """
tf.random.set_seed(42)
deep_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[12, X_train_seq.shape[2]]),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1)
])
fit_and_evaluate(deep_model, train_set, test_set, learning_rate=0.01)
"""
)

data = {
    "Modèle": ["RNN"],
    "Scaler": ["Robust"],
    "MAE": [0.583128],
    "RMSE": [0.872974],
    "R²": [0.270143],
}

st.dataframe(pd.DataFrame(data))

st.write(
    "Un simple RNN sur une seule couche donne de meilleurs résultats que plusieurs."
)

st.write("> TCN")

st.code(
    """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_2 (InputLayer)           │ (None, 12, 29)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ tcn_1 (TCN)                          │ (None, 128)                 │         557,184 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_6 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (Dense)                      │ (None, 32)                  │           4,128 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_5 (Dense)                      │ (None, 1)                   │              33 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘

 Total params: 561,345 (2.14 MB)

 Trainable params: 561,345 (2.14 MB)

"""
)

data = {
    "Modèle": ["TCN"],
    "Scaler": ["Robust"],
    "MAE": [2582.952873],
    "RMSE": [15004.88812],
    "R²": [0.161125],
}

st.dataframe(pd.DataFrame(data))


st.write(
    "Les résultats restent modestes. Pour améliorer "
    "la capture des fluctuations par nos modèles, notre "
    "dataset devrait être enrichi ou s'appuyer sur des séries "
    "temporelles plus étendues."
)

st.write("---")
st.write("### Axes d'amélioration")
st.write(
    "Nous pouvons encore enrichir notre dataset, "
    "en particulier avec des variables comme le trimestre, la saison et "
    "la zone géographique (province, Île-de-France). Cependant, après réflexion, "
    "il apparaît que l'inclusion du champ '*valeur foncière*' ne constitue pas "
    "une option optimale.  \n\n"
    "Avec plus de temps, il serait pertinent de revoir notre approche en "
    "l'excluant afin d'obtenir une analyse plus cohérente et pertinente."
)
