#!/usr/bin/env python3
"""
Model Demonstration Script
==========================
This script demonstrates all models from final.ipynb by:
1. Loading and preprocessing data
2. Training all classification and clustering models
3. Taking user input from terminal
4. Running predictions through all models
5. Displaying results

Models included:
- Classification: Logistic Regression, Decision Tree, LightGBM, CatBoost, Ensemble
- Clustering: K-Means, DBSCAN
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import sklearn components
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import loguniform, uniform

# Import imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Import LightGBM and CatBoost
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Skipping LightGBM model.")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Skipping CatBoost model.")


def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("=" * 70)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('online_shoppers_intention.csv')
    df_clean = df.copy()
    
    # Remove duplicates
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    after = len(df_clean)
    print(f"Removed {before - after} duplicate rows")
    
    # Outlier handling
    clip_01_cols = ['BounceRates', 'ExitRates', 'SpecialDay']
    for col in clip_01_cols:
        df_clean[col] = df_clean[col].clip(0, 1)
    
    clip_p99_cols = [
        'Administrative', 'Informational', 'ProductRelated',
        'Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration',
        'PageValues'
    ]
    for col in clip_p99_cols:
        p99 = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(upper=p99)
    
    # Encoding
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    visit_map = {'Other': 0, 'New Visitor': 1, 'Returning Visitor': 2}
    bool_map = {False: 0, True: 1}
    
    df_clean['Month'] = df_clean['Month'].map(month_map)
    df_clean['VisitorType'] = df_clean['VisitorType'].map(visit_map)
    df_clean['Weekend'] = df_clean['Weekend'].map(bool_map)
    df_clean['Revenue'] = df_clean['Revenue'].map(bool_map)
    
    print(f"Data shape: {df_clean.shape}")
    print("Preprocessing complete!\n")
    
    return df_clean


def train_classification_models(df_clean):
    """Train all classification models"""
    print("=" * 70)
    print("STEP 2: Training Classification Models")
    print("=" * 70)
    
    # Prepare data
    y = df_clean['Revenue'].astype(int)
    X = df_clean.drop(columns=['Revenue'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    
    # Preprocessor for imputation and scaling
    preprocessor = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocess training data first (impute and scale)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Apply SMOTE on preprocessed data (SMOTE doesn't accept NaN)
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_preprocessed, y_train)
    
    models = {}
    
    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    from sklearn.pipeline import Pipeline
    logreg_pipeline = Pipeline(steps=[
        ('classifier', LogisticRegression(
            max_iter=5000,
            random_state=42,
            solver="saga",
            penalty='elasticnet'
        ))
    ])
    
    param_dist_logreg = {
        'classifier__C': loguniform(1e-4, 1e2),
        'classifier__l1_ratio': uniform(0.0, 1.0)
    }
    
    gs_logreg = RandomizedSearchCV(
        logreg_pipeline,
        param_distributions=param_dist_logreg,
        n_iter=20,  # Reduced for faster execution
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    gs_logreg.fit(X_train_sm, y_train_sm)
    models['Logistic Regression'] = gs_logreg
    print(f"   Best params: {gs_logreg.best_params_}")
    
    # 2. Decision Tree
    print("\n2. Training Decision Tree...")
    from sklearn.pipeline import Pipeline
    dt_pipeline = Pipeline(steps=[
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    param_grid_dt = {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [None, 3, 5, 7, 10],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': [None, 'sqrt', 'log2']
    }
    
    gs_dt = RandomizedSearchCV(
        dt_pipeline,
        param_distributions=param_grid_dt,
        n_iter=30,  # Reduced for faster execution
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    gs_dt.fit(X_train_sm, y_train_sm)
    models['Decision Tree'] = gs_dt
    print(f"   Best params: {gs_dt.best_params_}")
    
    # 3. LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n3. Training LightGBM...")
        from sklearn.pipeline import Pipeline
        lgbm_pipeline = Pipeline(steps=[
            ('classifier', LGBMClassifier(objective="binary", n_estimators=200, random_state=42, verbose=-1))
        ])
        
        param_dist_lgbm = {
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 7],
            'classifier__num_leaves': [15, 31, 63],
            'classifier__min_child_samples': [10, 20, 30]
        }
        
        gs_lgbm = RandomizedSearchCV(
            lgbm_pipeline,
            param_distributions=param_dist_lgbm,
            n_iter=15,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        gs_lgbm.fit(X_train_sm, y_train_sm)
        models['LightGBM'] = gs_lgbm
        print(f"   Best params: {gs_lgbm.best_params_}")
    
    # 4. CatBoost
    if CATBOOST_AVAILABLE:
        print("\n4. Training CatBoost...")
        from sklearn.pipeline import Pipeline
        cat_pipeline = Pipeline(steps=[
            ('classifier', CatBoostClassifier(random_seed=42, auto_class_weights='Balanced', verbose=False))
        ])
        
        param_dist_cat = {
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__depth': [3, 5, 7],
            'classifier__l2_leaf_reg': [1, 3, 5]
        }
        
        gs_cat = RandomizedSearchCV(
            cat_pipeline,
            param_distributions=param_dist_cat,
            n_iter=15,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        gs_cat.fit(X_train_sm, y_train_sm)
        models['CatBoost'] = gs_cat
        print(f"   Best params: {gs_cat.best_params_}")
    
    print("\nClassification models trained successfully!")
    # Store preprocessor for later use in prediction
    models['_preprocessor'] = preprocessor
    return models, X_test_preprocessed, y_test


def train_clustering_models(df_clean):
    """Train clustering models"""
    print("\n" + "=" * 70)
    print("STEP 3: Training Clustering Models")
    print("=" * 70)
    
    # Prepare data
    X_cluster = df_clean.drop(columns=['Revenue']).copy()
    
    # Preprocessor
    preprocessor = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    X_cluster_scaled = preprocessor.fit_transform(X_cluster)
    
    models = {}
    preprocessors = {}
    
    # 1. K-Means
    print("\n1. Training K-Means...")
    silhouette_scores = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cluster_scaled)
        silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    models['K-Means'] = kmeans
    preprocessors['K-Means'] = preprocessor
    print(f"   Optimal k: {optimal_k}")
    print(f"   Silhouette Score: {silhouette_scores[np.argmax(silhouette_scores)]:.4f}")
    
    # 2. DBSCAN
    print("\n2. Training DBSCAN...")
    k = 4
    eps_values = np.arange(0.5, 3.0, 0.3)
    best_sil = -1
    best_eps = 1.5
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=k)
        labels = dbscan.fit_predict(X_cluster_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1 and n_noise < len(labels) * 0.3:
            mask = labels != -1
            if np.sum(mask) > 1:
                try:
                    sil_score = silhouette_score(X_cluster_scaled[mask], labels[mask])
                    if sil_score > best_sil:
                        best_sil = sil_score
                        best_eps = eps
                except:
                    pass
    
    dbscan = DBSCAN(eps=best_eps, min_samples=k)
    dbscan.fit(X_cluster_scaled)
    models['DBSCAN'] = dbscan
    preprocessors['DBSCAN'] = preprocessor
    print(f"   Optimal eps: {best_eps:.2f}")
    print(f"   Number of clusters: {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)}")
    if best_sil > 0:
        print(f"   Silhouette Score: {best_sil:.4f}")
    
    print("\nClustering models trained successfully!")
    return models, preprocessors


def get_user_input(use_sample=False):
    """Get input from user via terminal or use sample data"""
    print("\n" + "=" * 70)
    print("STEP 4: User Input")
    print("=" * 70)
    
    if use_sample:
        print("\nUsing sample data (high purchase probability example):\n")
        features = {
            'Administrative': 3.0,
            'Administrative_Duration': 100.0,
            'Informational': 2.0,
            'Informational_Duration': 50.0,
            'ProductRelated': 25.0,
            'ProductRelated_Duration': 1500.0,
            'BounceRates': 0.02,
            'ExitRates': 0.05,
            'PageValues': 50.0,
            'SpecialDay': 0.0,
            'Month': 11,
            'OperatingSystems': 2,
            'Browser': 2,
            'Region': 1,
            'TrafficType': 2,
            'VisitorType': 2,
            'Weekend': 0
        }
        print("Sample features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        return features
    
    print("\nPlease enter the following features for prediction:")
    print("(Press Enter to use sample data, or type 'sample' for sample input)\n")
    
    use_sample_input = input("Use sample data? (y/n, default=n): ").strip().lower()
    if use_sample_input in ['y', 'yes', 'sample']:
        return get_user_input(use_sample=True)
    
    features = {}
    
    # Numeric features
    print("\n--- Numeric Features ---")
    try:
        features['Administrative'] = float(input("Administrative (number of admin pages): ") or "0")
        features['Administrative_Duration'] = float(input("Administrative_Duration (seconds): ") or "0")
        features['Informational'] = float(input("Informational (number of info pages): ") or "0")
        features['Informational_Duration'] = float(input("Informational_Duration (seconds): ") or "0")
        features['ProductRelated'] = float(input("ProductRelated (number of product pages): ") or "0")
        features['ProductRelated_Duration'] = float(input("ProductRelated_Duration (seconds): ") or "0")
        features['BounceRates'] = float(input("BounceRates (0-1): ") or "0")
        features['ExitRates'] = float(input("ExitRates (0-1): ") or "0")
        features['PageValues'] = float(input("PageValues: ") or "0")
        features['SpecialDay'] = float(input("SpecialDay (0-1): ") or "0")
        
        # Categorical features
        print("\n--- Categorical Features ---")
        month_input = input("Month (1-12, Jan=1, Feb=2, ..., Dec=12): ") or "1"
        features['Month'] = int(month_input)
        
        features['OperatingSystems'] = int(input("OperatingSystems (1-8): ") or "1")
        features['Browser'] = int(input("Browser (1-13): ") or "1")
        features['Region'] = int(input("Region (1-9): ") or "1")
        features['TrafficType'] = int(input("TrafficType (1-20): ") or "1")
        
        visitor_input = input("VisitorType (0=Other, 1=New Visitor, 2=Returning Visitor): ") or "2"
        features['VisitorType'] = int(visitor_input)
        
        weekend_input = input("Weekend (0=False, 1=True): ") or "0"
        features['Weekend'] = int(weekend_input)
    except ValueError as e:
        print(f"\nError: Invalid input. Using sample data instead.")
        return get_user_input(use_sample=True)
    
    return features


def predict_classification(features_dict, models, X_test, y_test):
    """Run classification predictions"""
    print("\n" + "=" * 70)
    print("STEP 5: Classification Predictions")
    print("=" * 70)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([features_dict])
    
    # Get preprocessor
    preprocessor = models.get('_preprocessor')
    if preprocessor:
        input_preprocessed = preprocessor.transform(input_df)
    else:
        input_preprocessed = input_df
    
    results = {}
    
    for model_name, model in models.items():
        if model_name == '_preprocessor':
            continue
        try:
            # Predict
            prediction = model.predict(input_preprocessed)[0]
            probability = model.predict_proba(input_preprocessed)[0][1]
            
            results[model_name] = {
                'prediction': 'Purchase' if prediction == 1 else 'No Purchase',
                'probability': probability,
                'confidence': probability if prediction == 1 else (1 - probability)
            }
            
            print(f"\n{model_name}:")
            print(f"  Prediction: {results[model_name]['prediction']}")
            print(f"  Purchase Probability: {probability:.4f}")
            print(f"  Confidence: {results[model_name]['confidence']:.4f}")
            
        except Exception as e:
            print(f"\n{model_name}: Error - {str(e)}")
            results[model_name] = None
    
    # Ensemble prediction (if LightGBM and CatBoost available)
    if LIGHTGBM_AVAILABLE and CATBOOST_AVAILABLE and 'LightGBM' in results and 'CatBoost' in results:
        if results['LightGBM'] and results['CatBoost']:
            lgbm_prob = results['LightGBM']['probability']
            cat_prob = results['CatBoost']['probability']
            ensemble_prob = 0.6 * lgbm_prob + 0.4 * cat_prob
            ensemble_pred = 'Purchase' if ensemble_prob > 0.5 else 'No Purchase'
            
            print(f"\nEnsemble (Weighted Average):")
            print(f"  Prediction: {ensemble_pred}")
            print(f"  Purchase Probability: {ensemble_prob:.4f}")
            print(f"  Confidence: {ensemble_prob if ensemble_pred == 'Purchase' else (1 - ensemble_prob):.4f}")
            results['Ensemble'] = {
                'prediction': ensemble_pred,
                'probability': ensemble_prob,
                'confidence': ensemble_prob if ensemble_pred == 'Purchase' else (1 - ensemble_prob)
            }
    
    return results


def predict_clustering(features_dict, models, preprocessors):
    """Run clustering predictions"""
    print("\n" + "=" * 70)
    print("STEP 6: Clustering Assignments")
    print("=" * 70)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([features_dict])
    
    results = {}
    
    for model_name, model in models.items():
        try:
            preprocessor = preprocessors[model_name]
            input_scaled = preprocessor.transform(input_df)
            
            if model_name == 'K-Means':
                cluster = model.predict(input_scaled)[0]
                results[model_name] = {
                    'cluster': int(cluster),
                    'distance_to_center': np.linalg.norm(input_scaled[0] - model.cluster_centers_[cluster])
                }
                print(f"\n{model_name}:")
                print(f"  Assigned Cluster: {cluster}")
                print(f"  Distance to Cluster Center: {results[model_name]['distance_to_center']:.4f}")
                
            elif model_name == 'DBSCAN':
                # DBSCAN doesn't have predict method for new points
                # We need to check if the new point would be assigned to an existing cluster
                # by checking if it's within eps distance of any core point
                from sklearn.neighbors import NearestNeighbors
                
                # Get the training data that was used for DBSCAN
                # We'll use a simplified approach: find nearest labeled point
                print(f"\n{model_name}:")
                print(f"  Note: DBSCAN doesn't predict new points directly.")
                print(f"  To assign a new point, we check if it's within eps distance")
                print(f"  of any core point in an existing cluster.")
                
                # For demonstration, we'll use a simplified approach
                # In practice, you'd need to store the training data and core points
                eps = model.eps
                min_samples = model.min_samples
                
                # Create a temporary DBSCAN with the new point included
                # This is a simplified demonstration
                print(f"  eps parameter: {eps:.2f}")
                print(f"  min_samples: {min_samples}")
                print(f"  New point evaluation requires full dataset - showing parameters only")
                results[model_name] = {
                    'cluster': 'N/A (requires full dataset evaluation)',
                    'eps': eps,
                    'min_samples': min_samples
                }
                    
        except Exception as e:
            print(f"\n{model_name}: Error - {str(e)}")
            results[model_name] = None
    
    return results


def main():
    """Main function to run the demonstration"""
    print("\n" + "=" * 70)
    print("MODEL DEMONSTRATION SCRIPT")
    print("=" * 70)
    print("\nThis script demonstrates all models from final.ipynb")
    print("Models: Logistic Regression, Decision Tree, LightGBM, CatBoost, Ensemble")
    print("Clustering: K-Means, DBSCAN\n")
    
    import sys
    use_sample = '--sample' in sys.argv or '-s' in sys.argv
    
    # Load and preprocess data
    df_clean = load_and_preprocess_data()
    
    # Train models
    clf_models, X_test, y_test = train_classification_models(df_clean)
    cluster_models, cluster_preprocessors = train_clustering_models(df_clean)
    
    # Get user input
    features = get_user_input(use_sample=use_sample)
    
    # Run predictions
    clf_results = predict_classification(features, clf_models, X_test, y_test)
    cluster_results = predict_clustering(features, cluster_models, cluster_preprocessors)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nClassification Results:")
    for model_name, result in clf_results.items():
        if result:
            print(f"  {model_name}: {result['prediction']} (Probability: {result['probability']:.4f})")
    
    print("\nClustering Results:")
    for model_name, result in cluster_results.items():
        if result:
            if 'cluster' in result:
                if isinstance(result['cluster'], (int, str)):
                    print(f"  {model_name}: Cluster {result['cluster']}")
    
    print("\n" + "=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print("\nTip: Use --sample or -s flag to use sample data automatically")


if __name__ == "__main__":
    main()

