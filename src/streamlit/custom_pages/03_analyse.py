import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


IMG_FOLDER = "reports/figures/"


st.header("📊 Analyse des données")

st.write("")

st.subheader("Dataset Ventes 68 :")

sales_df = st.session_state["datasets"]["sales_df"]

num_cols = sales_df.select_dtypes(include="number").columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sales_df[num_cols])
scaled_df = pd.DataFrame(scaled_data, columns=num_cols)

fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=scaled_df)
plt.xticks(rotation=90)
plt.ylim(-2, 10)
plt.title("Boxplots normalisés (Z-score)")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(
    sales_df[
        [
            "prix_bien",
            "prix_maison",
            "prix_terrain",
            "surface",
            "nb_pieces",
            "prix_m2_vente",
        ]
    ].corr(),
    annot=True,
    cmap="coolwarm",
)

plt.title("Corrélations entre variables numériques (ventes)")
st.pyplot(fig)

st.write("---")

st.subheader("Dataset Locations 68 :")

rentals_df = st.session_state["datasets"]["rentals_df"]

num_cols = rentals_df.select_dtypes(include="number").columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(rentals_df[num_cols])
scaled_df = pd.DataFrame(scaled_data, columns=num_cols)

fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=scaled_df)
plt.xticks(rotation=90)
plt.ylim(-2, 10)
plt.title("Boxplots normalisés (Z-score)")
st.pyplot(fig)

plt.title("Corrélations entre variables numériques (locations)")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(
    rentals_df[["etage", "nb_pieces", "surface", "balcon", "nb_terraces"]].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Corrélations entre variables numériques (locations)", fontsize=16)
st.pyplot(fig)

sales_df["year"] = pd.to_datetime(sales_df["date"]).dt.year
sales_years_df = sales_df.groupby("year").size().reset_index(name="count")

rentals_df["year"] = pd.to_datetime(rentals_df["date"]).dt.year
rentals_years_df = rentals_df.groupby("year").size().reset_index(name="count")


fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.barplot(data=sales_years_df, x="year", y="count", hue="year", ax=ax[0])
ax[0].set_title("Biens en ventes par année", fontsize=16)
ax[0].set_xlabel("Année", fontsize=14)
ax[0].set_ylabel("Nombre de biens", fontsize=14)
ax[0].tick_params(axis="x", rotation=45)
ax[0].legend(title="Année")

sns.barplot(data=rentals_years_df, x="year", y="count", hue="year", ax=ax[1])
ax[1].set_title("Biens en locations par année", fontsize=16)
ax[1].set_xlabel("Année", fontsize=14)
ax[1].set_ylabel("Nombre de biens", fontsize=14)
ax[1].tick_params(axis="x", rotation=45)
ax[1].legend(title="Année")

plt.tight_layout()
st.pyplot(fig)

st.write("---")

st.subheader("Dataset Valeurs foncières :")
# optimisation ressources computationnelles (2024 seulement)
output_path = "data/raw/full_2024.csv.gz"
dvf_df_2024 = pd.read_csv(output_path, low_memory=False, nrows=50000)
num_cols = dvf_df_2024.select_dtypes(include=["number"])
cat_cols = dvf_df_2024.select_dtypes(include=["object"])

fig, ax = plt.subplots(figsize=(20, 12))
sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm")
plt.title("Corrélations entre variables numériques (data-gouv)", fontsize=16)
st.pyplot(fig)

st.image(f"{IMG_FOLDER}evolution_des_prix.jpg", use_container_width=True)

st.image(f"{IMG_FOLDER}repartition_prix_median.jpg", use_container_width=True)
