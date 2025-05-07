import lightgbm
from preprocessing import *
from classification import *
import pandas as pd
import xgboost
import pickle
import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    xgboost.config_context(verbosity=0)
    
    # Comment code below if you want to test all models in one run
    print("""Available models:
            
            'logistic': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'decision_tree': DecisionTreeClassifier,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier

    """)
    model = input('Enter which model you want to evaluate:')
    

    data = pd.read_excel('problem_data_new.xlsx')
    x_train, x_test, y_train, y_test = modify_eliminate_cols(data)
    x_train, x_test = impute(x_train=x_train, x_test=x_test)
    x_train, x_test = encode(x_train=x_train, x_test=x_test, y_train=y_train)
    
    pipeline = build_pipeline(shap_features=10)
    pipeline.fit(x_train, y_train)
    
    shap_selected = pipeline.named_steps['shap_selector'].get_support()
    rfe_selected = pipeline.named_steps['rfe_selector'].get_support()
    print(f"SHAP top features: {shap_selected}")
    print(f"RFE-selected features: {rfe_selected}")
    
    # Results are the same when you simply use the top 5 SHAP features
    # Could be useful if there is a difference between the two
    # final_features = list(set(shap_selected[:5]).union(set(rfe_selected[:5])))
    
    final_features = shap_selected[:5]
    x_train_reduced = x_train[final_features]
    x_test_reduced = x_test[final_features]

    # If you want to reuse the best parameters, uncomment the code below
    # with open('config/best_params_logistic.pkl', 'rb') as f:
    #     best_params = pickle.load(f)

    # Uncomment code below if you want to test all models in one run
    # models = ['logistic', 'random_forest', 'gradient_boosting', 'decision_tree', 'xgboost', 'lightgbm']
    # tuner = ModelTuner(X_train=x_train_reduced, X_test=x_test_reduced, y_train=y_train, y_test=y_test)
    # for model in models:
    #         results, best_params = tuner.run_model(model, n_trials=10)

    #         with open(f'config/best_params_{model}.pkl', 'wb') as f:
    #             pickle.dump(best_params, f)

    tuner = ModelTuner(X_train=x_train_reduced, X_test=x_test_reduced, y_train=y_train, y_test=y_test)
    results, best_params = tuner.run_model(model, n_trials=10)

    with open(f'config/best_params_{model}.pkl', 'wb') as f:
        pickle.dump(best_params, f)