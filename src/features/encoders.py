from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        self.ohe.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        ohe_df = pd.DataFrame(self.ohe.transform(X[self.columns]),
                              columns=self.ohe.get_feature_names_out(
                                  self.columns),
                              index=X.index)
        X = X.drop(columns=self.columns)
        return pd.concat([X, ohe_df], axis=1)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.freq_maps = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.freq_maps[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            X_transformed[col] = X_transformed[col].map(
                self.freq_maps[col]).fillna(0).astype(int)
        return X_transformed


class BinaryLotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, lot_cols):
        self.lot_cols = lot_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.lot_cols:
            X_transformed[col] = (X_transformed[col] != 0).astype(int)
        return X_transformed


class ImprovedGeoEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=50, random_state=42, handle_missing='mean'):
        """
        Encoder géographique amélioré avec gestion robuste des NaN
        
        Parameters:
        -----------
        n_clusters : int
            Nombre de clusters géographiques
        random_state : int
            Graine aléatoire pour la reproductibilité
        handle_missing : str
            Stratégie pour gérer les valeurs manquantes ('mean', 'median', 'zero')
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.handle_missing = handle_missing
        self.commune_price_map = {}
        self.dept_price_map = {}
        self.cluster_price_map = {}
        self.kmeans = None
        self.coord_scaler = StandardScaler()
        self.is_fitted = False
        
        # Valeurs de fallback
        self.commune_fallback = 0
        self.dept_fallback = 0
        self.cluster_fallback = 0
        
    def _calculate_fallback_values(self, price_maps):
        """
        Calcule les valeurs de fallback selon la stratégie choisie
        """
        if self.handle_missing == 'mean':
            self.commune_fallback = np.mean(list(self.commune_price_map.values())) if self.commune_price_map else 0
            self.dept_fallback = np.mean(list(self.dept_price_map.values())) if self.dept_price_map else 0
            self.cluster_fallback = np.mean(list(self.cluster_price_map.values())) if self.cluster_price_map else 0
        elif self.handle_missing == 'median':
            self.commune_fallback = np.median(list(self.commune_price_map.values())) if self.commune_price_map else 0
            self.dept_fallback = np.median(list(self.dept_price_map.values())) if self.dept_price_map else 0
            self.cluster_fallback = np.median(list(self.cluster_price_map.values())) if self.cluster_price_map else 0
        else:  # 'zero'
            self.commune_fallback = 0
            self.dept_fallback = 0
            self.cluster_fallback = 0
    
    def fit(self, X, y=None):
        """
        Crée des encodings basés sur les prix moyens par localisation
        
        Parameters:
        -----------
        X : DataFrame
            Features d'entraînement
        y : Series
            Target (prix_m2_vente)
        """
        if y is None:
            raise ValueError("y (target) est requis pour l'entraînement de l'encoder géographique")
        
        # Vérifier l'alignement des index
        if not X.index.equals(y.index):
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
        
        # Créer une copie de travail
        X_work = X.copy()
        y_work = y.copy()
        
        # Reset des index pour éviter les problèmes
        X_work = X_work.reset_index(drop=True)
        y_work = y_work.reset_index(drop=True)
        
        # Prix moyen par commune
        temp_df = pd.DataFrame({
            'code_commune': X_work['code_commune'],
            'target': y_work
        })
        commune_prices = temp_df.groupby('code_commune')['target'].mean()
        self.commune_price_map = commune_prices.to_dict()
        
        # Prix moyen par département
        temp_df['code_departement'] = X_work['code_departement']
        dept_prices = temp_df.groupby('code_departement')['target'].mean()
        self.dept_price_map = dept_prices.to_dict()
        
        # Clustering géographique si latitude/longitude disponibles
        if 'latitude' in X_work.columns and 'longitude' in X_work.columns:
            # Filtrer les coordonnées valides
            coords_mask = (
                X_work['latitude'].notna() & 
                X_work['longitude'].notna() &
                (X_work['latitude'] != 0) & 
                (X_work['longitude'] != 0) &
                (X_work['latitude'].between(-90, 90)) &
                (X_work['longitude'].between(-180, 180))
            )
            
            valid_coords_count = coords_mask.sum()
            
            if valid_coords_count >= self.n_clusters:
                coords = X_work.loc[coords_mask, ['latitude', 'longitude']].copy()
                target_coords = y_work.loc[coords_mask].copy()
                
                # Normaliser les coordonnées
                coords_scaled = self.coord_scaler.fit_transform(coords)
                
                # Ajuster le nombre de clusters
                n_clusters_actual = min(self.n_clusters, len(coords))
                
                # Clustering
                self.kmeans = KMeans(
                    n_clusters=n_clusters_actual, 
                    random_state=self.random_state, 
                    n_init=10
                )
                clusters = self.kmeans.fit_predict(coords_scaled)
                
                # Prix moyen par cluster géographique
                cluster_df = pd.DataFrame({
                    'cluster': clusters,
                    'target': target_coords.values
                })
                cluster_prices = cluster_df.groupby('cluster')['target'].mean()
                self.cluster_price_map = cluster_prices.to_dict()
                
                print(f"{n_clusters_actual} clusters géographiques créés")
            else:
                print(f"Pas assez de coordonnées valides ({valid_coords_count}) pour créer {self.n_clusters} clusters")
                self.kmeans = None
        else:
            print("Pas de colonnes latitude/longitude trouvées")
            self.kmeans = None
        
        # Calculer les valeurs de fallback
        self._calculate_fallback_values(None)
        
        self.is_fitted = True
        
        return self
    
    def transform(self, X):
        """
        Applique les encodings géographiques avec gestion robuste des NaN
        
        Parameters:
        -----------
        X : DataFrame
            Features à transformer
            
        Returns:
        --------
        DataFrame
            Features avec encodings géographiques ajoutés (sans NaN)
        """
        if not self.is_fitted:
            raise ValueError("L'encoder doit être entraîné avant la transformation. Appelez fit() d'abord.")
        
        X_encoded = X.copy()
        
        # 1. Encoding par prix moyen de commune (avec gestion robuste des NaN)
        X_encoded['commune_price_encoding'] = X_encoded['code_commune'].map(
            self.commune_price_map
        ).fillna(self.commune_fallback)
        
        # Vérification et correction des NaN restants
        if X_encoded['commune_price_encoding'].isna().any():
            print("NaN détectés dans commune_price_encoding, correction...")
            X_encoded['commune_price_encoding'] = X_encoded['commune_price_encoding'].fillna(self.commune_fallback)
        
        # 2. Encoding par prix moyen de département
        X_encoded['dept_price_encoding'] = X_encoded['code_departement'].map(
            self.dept_price_map
        ).fillna(self.dept_fallback)
        
        # Vérification et correction des NaN restants
        if X_encoded['dept_price_encoding'].isna().any():
            print("NaN détectés dans dept_price_encoding, correction...")
            X_encoded['dept_price_encoding'] = X_encoded['dept_price_encoding'].fillna(self.dept_fallback)
        
        # 3. Clustering géographique avec gestion robuste
        if self.kmeans is not None and 'latitude' in X.columns and 'longitude' in X.columns:
            try:
                # Préparer les coordonnées
                coords = X_encoded[['latitude', 'longitude']].copy()
                
                # Gérer les valeurs manquantes dans les coordonnées
                coords_mask = (
                    coords['latitude'].notna() & 
                    coords['longitude'].notna() &
                    (coords['latitude'] != 0) & 
                    (coords['longitude'] != 0) &
                    (coords['latitude'].between(-90, 90)) &
                    (coords['longitude'].between(-180, 180))
                )
                
                # Initialiser les colonnes avec des valeurs par défaut
                X_encoded['geo_cluster'] = -1  # -1 pour les coordonnées invalides
                X_encoded['cluster_price_encoding'] = self.cluster_fallback
                
                if coords_mask.sum() > 0:
                    # Coordonnées valides
                    valid_coords = coords.loc[coords_mask]
                    
                    # Normaliser avec le scaler entraîné
                    coords_scaled = self.coord_scaler.transform(valid_coords)
                    
                    # Prédire les clusters
                    clusters = self.kmeans.predict(coords_scaled)
                    
                    # Assigner les clusters
                    X_encoded.loc[coords_mask, 'geo_cluster'] = clusters
                    
                    # Assigner les prix de cluster avec gestion des NaN
                    cluster_prices = pd.Series(clusters).map(self.cluster_price_map)
                    cluster_prices = cluster_prices.fillna(self.cluster_fallback)
                    
                    X_encoded.loc[coords_mask, 'cluster_price_encoding'] = cluster_prices.values
                
                # Vérification finale des NaN
                if X_encoded['cluster_price_encoding'].isna().any():
                    print("NaN détectés dans cluster_price_encoding, correction finale...")
                    X_encoded['cluster_price_encoding'] = X_encoded['cluster_price_encoding'].fillna(self.cluster_fallback)
                
            except Exception as e:
                print(f"Erreur lors du clustering géographique : {e}")
                X_encoded['geo_cluster'] = -1
                X_encoded['cluster_price_encoding'] = self.cluster_fallback
        else:
            # Si pas de clustering disponible
            X_encoded['geo_cluster'] = -1
            X_encoded['cluster_price_encoding'] = self.cluster_fallback
        
        # 4. Features dérivées (sans NaN)
        X_encoded['commune_dept_ratio'] = (
            X_encoded['commune_price_encoding'] / 
            (X_encoded['dept_price_encoding'] + 1e-8)  # Éviter division par zéro
        )
        
        # Indicateur de rareté (basé sur la fréquence des communes)
        commune_counts = X_encoded['code_commune'].value_counts()
        X_encoded['commune_frequency'] = X_encoded['code_commune'].map(commune_counts).fillna(1)
        X_encoded['commune_rarity'] = 1 / (X_encoded['commune_frequency'] + 1)
        
        # 5. Vérification finale : s'assurer qu'il n'y a pas de NaN
        numeric_cols = ['commune_price_encoding', 'dept_price_encoding', 'cluster_price_encoding', 
                       'commune_dept_ratio', 'commune_frequency', 'commune_rarity']
        
        for col in numeric_cols:
            if col in X_encoded.columns:
                nan_count = X_encoded[col].isna().sum()
                if nan_count > 0:
                    print(f"{nan_count} NaN trouvés dans {col}, correction...")
                    if col in ['commune_price_encoding', 'dept_price_encoding', 'cluster_price_encoding']:
                        X_encoded[col] = X_encoded[col].fillna(self.cluster_fallback)
                    else:
                        X_encoded[col] = X_encoded[col].fillna(0)
        
        # Vérification finale
        total_nans = X_encoded[numeric_cols].isna().sum().sum()
        if total_nans > 0:
            print(f"{total_nans} NaN restants détectés, correction forcée...")
            X_encoded = X_encoded.fillna(0)
        
        return X_encoded
    
    def fit_transform(self, X, y=None):
        """
        Entraîne l'encoder et transforme les données en une seule étape
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """
        Retourne les noms des features créées par l'encoder
        """
        return [
            'commune_price_encoding',
            'dept_price_encoding', 
            'geo_cluster',
            'cluster_price_encoding',
            'commune_dept_ratio',
            'commune_frequency',
            'commune_rarity'
        ]

# Fonction de test pour vérifier la robustesse
def test_robust_geo_encoder():
    """
    Test complet avec gestion des cas difficiles
    """
    # Créer des données de test avec des cas difficiles
    np.random.seed(42)
    n_samples = 1000
    
    # Données avec des NaN et des valeurs extrêmes
    test_data = pd.DataFrame({
        'code_commune': np.random.choice([75001, 75002, 75016, 92100, 93200, 99999], n_samples),  # Code inexistant
        'code_departement': np.random.choice([75, 92, 93, 99], n_samples),  # Département inexistant
        'latitude': np.random.normal(48.8566, 0.1, n_samples),
        'longitude': np.random.normal(2.3522, 0.1, n_samples),
    })
    
    # Introduire des NaN dans les coordonnées
    nan_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
    test_data.loc[nan_indices, 'latitude'] = np.nan
    test_data.loc[nan_indices[:50], 'longitude'] = np.nan
    
    # Target avec quelques valeurs extrêmes
    target = pd.Series(
        np.random.normal(8000, 2000, n_samples),
        name='prix_m2_vente'
    )
    
    # Quelques valeurs extrêmes dans le target
    target.iloc[:10] = np.random.normal(50000, 5000, 10)  # Très cher
    target.iloc[-10:] = np.random.normal(1000, 200, 10)   # Prix très bas
    
    # Test de l'encoder
    encoder = ImprovedGeoEncoder(n_clusters=20, handle_missing='mean')
    
    try:
        # Test fit_transform
        X_transformed = encoder.fit_transform(test_data, target)
        
        # Vérifier qu'il n'y a pas de NaN
        nan_counts = X_transformed.select_dtypes(include=[np.number]).isna().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            print(f"{total_nans} NaN trouvés:")
        
        # Test transform séparé
        X_test = test_data.iloc[:100].copy()
        X_test_transformed = encoder.transform(X_test)
        
        # Vérifier la compatibilité avec StandardScaler
        scaler = StandardScaler()
        
        numeric_features = X_transformed.select_dtypes(include=[np.number]).columns
        X_scaled = scaler.fit_transform(X_transformed[numeric_features])
        
        # Afficher les statistiques des nouvelles features
        new_features = encoder.get_feature_names_out()
        for feature in new_features:
            if feature in X_transformed.columns:
                col_data = X_transformed[feature]
                print(f"   {feature:25}: {col_data.mean():8.2f} ± {col_data.std():6.2f} (min: {col_data.min():6.2f}, max: {col_data.max():6.2f})")
        
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_robust_geo_encoder()
    def __init__(self, n_clusters=50, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.commune_price_map = {}
        self.dept_price_map = {}
        self.cluster_price_map = {}
        self.kmeans = None
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """
        Crée des encodings basés sur les prix moyens par localisation
        
        Parameters:
        -----------
        X : DataFrame
            Features d'entraînement
        y : Series
            Target (prix_m2_vente)
        """
        if y is None:
            raise ValueError("y (target) est requis pour l'entraînement de l'encoder géographique")
        if X.index.duplicated().any():
            raise ValueError("X contient des index dupliqués, ce qui empêche le merge avec y")

        y_aligned = y.loc[X.index]
        X_with_target = X
        X_with_target['target'] = y_aligned
        
        # Prix moyen par commune
        commune_prices = X_with_target.groupby('code_commune')['target'].mean()
        self.commune_price_map = commune_prices.to_dict()
        
        # Prix moyen par département
        dept_prices = X_with_target.groupby('code_departement')['target'].mean()
        self.dept_price_map = dept_prices.to_dict()
        
        # Clustering géographique si latitude/longitude disponibles
        if 'latitude' in X.columns and 'longitude' in X.columns:
            coords_mask = X[['latitude', 'longitude']].notna().all(axis=1)
            if coords_mask.sum() > 0:
                coords = X.loc[coords_mask, ['latitude', 'longitude']]
                
                # Ajuster le nombre de clusters si nécessaire
                n_clusters = min(self.n_clusters, len(coords))
                
                self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
                clusters = self.kmeans.fit_predict(coords)
                
                # Prix moyen par cluster géographique
                coords_with_clusters = coords.copy()
                coords_with_clusters['cluster'] = clusters
                coords_with_clusters['target'] = y.loc[coords.index]
                self.cluster_price_map = coords_with_clusters.groupby('cluster')['target'].mean().to_dict()
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Applique les encodings géographiques
        
        Parameters:
        -----------
        X : DataFrame
            Features à transformer
            
        Returns:
        --------
        DataFrame
            Features avec encodings géographiques ajoutés
        """
        if not self.is_fitted:
            raise ValueError("L'encoder doit être entraîné avant la transformation. Appelez fit() d'abord.")
        
        X_encoded = X.copy()
        
        # Encoding par prix moyen de commune
        commune_mean = np.mean(list(self.commune_price_map.values())) if self.commune_price_map else 0
        X_encoded['commune_price_encoding'] = X_encoded['code_commune'].map(
            self.commune_price_map
        ).fillna(commune_mean)
        
        # Encoding par prix moyen de département
        dept_mean = np.mean(list(self.dept_price_map.values())) if self.dept_price_map else 0
        X_encoded['dept_price_encoding'] = X_encoded['code_departement'].map(
            self.dept_price_map
        ).fillna(dept_mean)
        
        # Clustering géographique
        if self.kmeans is not None and 'latitude' in X.columns and 'longitude' in X.columns:
            coords = X[['latitude', 'longitude']].copy()
            
            # Gérer les valeurs manquantes
            coords_filled = coords.fillna(coords.mean())
            
            try:
                clusters = self.kmeans.predict(coords_filled)
                X_encoded['geo_cluster'] = clusters
                
                # Prix moyen par cluster
                cluster_mean = np.mean(list(self.cluster_price_map.values())) if self.cluster_price_map else 0
                X_encoded['cluster_price_encoding'] = pd.Series(clusters).map(
                    self.cluster_price_map
                ).fillna(cluster_mean)
                
            except Exception as e:
                print(f"Erreur lors du clustering géographique : {e}")
                X_encoded['geo_cluster'] = 0
                X_encoded['cluster_price_encoding'] = dept_mean
        else:
            # Si pas de coordonnées, utiliser des valeurs par défaut
            X_encoded['geo_cluster'] = 0
            X_encoded['cluster_price_encoding'] = dept_mean
      
        X_encoded['cluster_price_encoding'].fillna(0)
        return X_encoded
    
    def fit_transform(self, X, y=None):
        """
        Entraîne l'encoder et transforme les données en une seule étape
        
        Parameters:
        -----------
        X : DataFrame
            Features d'entraînement
        y : Series
            Target (prix_m2_vente)
            
        Returns:
        --------
        DataFrame
            Features transformées avec encodings géographiques
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """
        Retourne les noms des features de sortie
        """
        if input_features is None:
            input_features = []
        
        output_features = list(input_features) + [
            'commune_price_encoding',
            'dept_price_encoding', 
            'geo_cluster',
            'cluster_price_encoding'
        ]
        
        return output_features