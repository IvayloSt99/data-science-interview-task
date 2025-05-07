import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import shap
import optuna
from time import time
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

""" The classes below use hyperparameter tuning during the feature selection phase. 
Hyperparameter tuning may not be completely justified here, since I managed to achieve almost the same results using more or less arbitrary params,
but I'll leave it as it is, so that the code can be reused on new data. """

class SHAPFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select top-N features based on SHAP importance from an XGBoost model.
    Automatically shows a SHAP summary plot during fitting.
    Added Optuna for opimization
    """
    def __init__(self, n_features=20, model_params=None, tune_params=True, 
                 n_trials=10, scoring='roc_auc', cv=3):
        self.n_features = n_features
        self.model_params = model_params or {}
        self.tune_params = tune_params
        self.n_trials = n_trials
        self.scoring = scoring
        self.cv = cv
        self.selected_features_ = []
        self.explainer_ = None
        self.shap_values_ = None
        self.best_params_ = None
        self.study_ = None

    def _objective(self, trial, X, y):
        """ Optuna objective function for hyperparameter optimization """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }

        # Add any fixed parameters that were provided
        params.update(self.model_params)
        
        model = LGBMClassifier(**params, random_state=42, verbosity=-1)
        score = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring).mean()
        return score

    def fit(self, X, y):
        # Perform hyperparameter tuning (True by default)
        if self.tune_params:
            print(f"\n[SHAPFeatureSelector] Starting hyperparameter optimization with {self.n_trials} trials...")
            self.study_ = optuna.create_study(direction='maximize')
            self.study_.optimize(lambda trial: self._objective(trial, X, y), 
                               n_trials=self.n_trials)
            
            self.best_params_ = self.study_.best_params
            print(f"[SHAPFeatureSelector] Best parameters found: {self.best_params_}")
            print(f"[SHAPFeatureSelector] Best {self.scoring} score: {self.study_.best_value:.4f}")
            
            # Update model_params with the best parameters found
            final_params = self.best_params_.copy()
        else:
            final_params = self.model_params.copy()
        
        # Add any fixed parameters that weren't tuned
        final_params.update({k: v for k, v in self.model_params.items() 
                            if k not in final_params})
        
        # Train final model with selected parameters
        # model = LGBMClassifier(**final_params, random_state=42, verbosity=-1)
        model = XGBClassifier(**self.model_params, seed=42)
        model.fit(X, y)

        # Compute SHAP values
        self.explainer_ = shap.Explainer(model, X)
        self.shap_values_ = self.explainer_(X)

        # Compute mean absolute SHAP values per feature
        shap_importance = np.abs(self.shap_values_.values).mean(axis=0)
        feature_ranks = pd.Series(shap_importance, index=X.columns).sort_values(ascending=False)
        self.selected_features_ = feature_ranks.head(self.n_features).index.tolist()

        # Automatically show SHAP summary plot
        print("\n[SHAPFeatureSelector] SHAP summary plot for top features:")
        shap.summary_plot(self.shap_values_, X, show=True)

        return self

    def transform(self, X):
        return X[self.selected_features_]

    def get_support(self):
        return self.selected_features_

class RFEFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Apply RFECV on the SHAP-selected features using XGBoost.
    Automatically selects optimal number of features via cross-validation.
    """
    def __init__(self, cv=5, n_trials=10, tune_params=True, scoring='accuracy', model_params=None):
        self.cv = cv
        self.n_trials = n_trials
        self.tune_params = tune_params
        self.scoring = scoring
        self.model_params = model_params or {}
        self.selected_features_ = []
        self.rfecv_ = None
        self.best_params_ = None
        self.study_ = None

    def _objective(self, trial, X, y):
        """ Optuna objective function for hyperparameter optimization """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }

        # Add any fixed parameters that were provided
        params.update(self.model_params)
        model = XGBClassifier(**self.model_params, seed=42)
        # model = LGBMClassifier(**params, random_state=42, verbosity=-1)
        score = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring).mean()
        return score

    def fit(self, X, y):
                # Perform hyperparameter tuning (True by default)
        if self.tune_params:
            print(f"\n[RFEFeatureSelector] Starting hyperparameter optimization with {self.n_trials} trials...")
            self.study_ = optuna.create_study(direction='maximize')
            self.study_.optimize(lambda trial: self._objective(trial, X, y), 
                               n_trials=self.n_trials)
            
            self.best_params_ = self.study_.best_params
            print(f"[RFEFeatureSelector] Best parameters found: {self.best_params_}")
            print(f"[RFEFeatureSelector] Best {self.scoring} score: {self.study_.best_value:.4f}")
            
            # Update model_params with the best parameters found
            final_params = self.best_params_.copy()
        else:
            final_params = self.model_params.copy()
        
        # Add any fixed parameters that weren't tuned
        final_params.update({k: v for k, v in self.model_params.items() 
                            if k not in final_params})
        
        model = XGBClassifier(**self.model_params, seed=42)
        # model=LGBMClassifier(**final_params, random_state=42, verbosity=-1)
        cv_strategy = StratifiedKFold(n_splits=self.cv)
        self.rfecv_ = RFECV(estimator=model, step=1, cv=cv_strategy, scoring=self.scoring)
        self.rfecv_.fit(X, y)
        self.selected_features_ = X.columns[self.rfecv_.support_].tolist()
        print(f"[RFEFeatureSelector] Optimal number of features selected: {len(self.selected_features_)}")
        return self

    def transform(self, X):
        return X[self.selected_features_]

    def get_support(self):
        return self.selected_features_

def build_pipeline(model_params=None, shap_features=20):
    
    pipeline = Pipeline([
        # Originally, I intended to integrate the various preprocessing steps here.
        # ('preprocessing', preprocess),
        ('shap_selector', SHAPFeatureSelector(n_features=shap_features, model_params=model_params)),
        ('rfe_selector', RFEFeatureSelector(model_params=model_params))
    ])

    return pipeline

class ModelTuner:
    def __init__(self, X_train, X_test, y_train, y_test, model_params=None):
        """
        Initialize the ModelTuner with data and basic configurations.
        
        Args:
            X_(train | test)(pd.DataFrame): Features dataframe
            y_(train | test) (pd.Series): Target variable
            test_size (float): Proportion of test set
        """

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_params = model_params or {}

        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Store results
        self.results = pd.DataFrame(columns=['Model', 'Accuracy', 'Recall', 
                                            'Precision', 'F1', 'ROC AUC', 
                                            'PR AUC', 'Training Time (s)',
                                            'Best Params'])
        self.models = {}
    
    def evaluate_model(self, model, model_name, params=None):
        """
        Evaluate a model and store the results.
        
        Args:
            model: Trained model instance
            model_name (str): Name of the model
            params (dict): Best parameters found by Optuna
        """
        start_time = time()
        
        # Train model
        model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test_scaled)
        y_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_proba) if y_proba is not None else np.nan
        pr_auc = average_precision_score(self.y_test, y_proba) if y_proba is not None else np.nan
        
        training_time = time() - start_time
        
        # Store results
        result_row = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'F1': f1,
            'ROC AUC': roc_auc,
            'PR AUC': pr_auc,
            'Training Time (s)': training_time,
            'Best Params': str(params)
        }
        
        # Store model
        self.models[model_name] = model
        
        # Print evaluation results
        print(f"\n{'='*50}")
        print(f"Evaluation results for {model_name}:")
        print('='*50)
        print(classification_report(self.y_test, y_pred))
        print(f"ROC AUC: {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC AUC: Not available")
        print(f"PR AUC: {pr_auc:.4f}" if not np.isnan(pr_auc) else "PR AUC: Not available")
        print(f"Training Time: {training_time:.2f} seconds")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_pred, model_name)
        
        return pd.DataFrame([result_row])
    
    def plot_confusion_matrix(self, y_pred, model_name):
        """ Plot confusion matrix for a model's predictions. """
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()
    
    def get_model_objective(self, model_name):
        """ Return the appropriate objective function for the given model. """
        objectives = {
            'logistic': self._objective_logistic,
            'random_forest': self._objective_rf,
            'gradient_boosting': self._objective_gb,
            'decision_tree': self._objective_dt,
            'xgboost': self._objective_xgb,
            'lightgbm': self._objective_lgbm
        }
        
        model_map = {
            'logistic': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'decision_tree': 'Decision Tree',
            'xgboost': 'XGBoost',
            'lightgbm': 'LGBMClassifier'
        }
        
        if model_name not in objectives:
            raise ValueError(f"Unknown model: {model_name}. Available options: {list(model_map.keys())}")
            
        return objectives[model_name], model_map[model_name]
    
    def create_model(self, model_name, params):
        """ Instantiate a model with given parameters, filtering out irrelevant ones. """
        model_classes = {
            'logistic': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'decision_tree': DecisionTreeClassifier,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier
        }
        
        # Get the appropriate model class
        model_class = model_classes[model_name]
        
        # Filter parameters to only include valid ones for this model
        valid_params = {}
        for param, value in params.items():
            # Check if parameter exists in the model's __init__
            if param in model_class().get_params():
                valid_params[param] = value
        
        # Create model with filtered parameters
        model = model_class(**valid_params, random_state=42)
        
        # Special handling for XGBoost
        if model_name == 'xgboost':
            model.set_params(eval_metric='logloss', use_label_encoder=False)
        
        return model

    def run_model(self, model_name, n_trials=15):
        """
        Run hyperparameter tuning and evaluation for a single model.
        
        Args:
            model_name (str): Name of model to run (see get_model_objective for options)
            n_trials (int): Number of Optuna trials
            
        Returns:
            pd.DataFrame: Results dataframe with evaluation metrics
        """
        # Get the appropriate objective function and display name
        objective_func, display_name = self.get_model_objective(model_name)
        
        print(f"\n{'='*50}")
        print(f"Tuning {display_name} with Optuna ({n_trials} trials)")
        print('='*50)
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective_func, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        print(f"Best F1-score: {study.best_value:.4f}")
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Create and evaluate model with best parameters
        model = self.create_model(model_name, best_params)
        result_df = self.evaluate_model(model, display_name, best_params)
        
        # Update overall results
        self.results = pd.concat([self.results, result_df], ignore_index=True)
        
        return result_df, best_params
    
    # Objective functions for each model type
    def _objective_logistic(self, trial):

        params = {
            'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }

        model = LogisticRegression(**params, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        return f1_score(self.y_test, y_pred)
    
    def _objective_rf(self, trial):

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        return f1_score(self.y_test, y_pred)
    
    def _objective_gb(self, trial):

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        return f1_score(self.y_test, y_pred)
    
    def _objective_dt(self, trial):

        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }

        model = DecisionTreeClassifier(**params, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        return f1_score(self.y_test, y_pred)
    
    def _objective_xgb(self, trial):

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }

        model = XGBClassifier(**params, random_state=42, eval_metric='logloss', use_label_encoder=False)
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        return f1_score(self.y_test, y_pred)

    def _objective_lgbm(self, trial):

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }

        model = LGBMClassifier(**params, random_state=42, verbosity=-1)
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        return f1_score(self.y_test, y_pred)
