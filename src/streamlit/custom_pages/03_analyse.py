import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import os
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components
import requests
from io import StringIOimport streamlit as st


IMG_FOLDER = "reports/figures/"

st.header("📊 Analyse des données")

st.write("")

st.subheader("Dataset Ventes 68 :")

st.image(f"{IMG_FOLDER}boxplot_ventes_68.jpg", use_container_width=True)

st.image(f"{IMG_FOLDER}heatmap_ventes_68.jpg", use_container_width=True)

st.write("---")

st.subheader("Dataset Locations 68 :")

st.image(f"{IMG_FOLDER}boxplot_location_68.jpg", use_container_width=True)

st.image(f"{IMG_FOLDER}heatmap_location_68.jpg", use_container_width=True)

st.image(f"{IMG_FOLDER}evolution_ventes_loc_68.jpg", use_container_width=True)

st.write("---")

st.subheader("Dataset Valeurs foncières :")

st.image(f"{IMG_FOLDER}heatmap_data_gouv.jpg", use_container_width=True)

st.image(f"{IMG_FOLDER}evolution_des_prix.jpg", use_container_width=True)

st.image(f"{IMG_FOLDER}repartition_prix_median.jpg", use_container_width=True)
