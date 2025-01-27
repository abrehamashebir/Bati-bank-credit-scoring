import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class ModelTraining:
    def __init__(self, data):
        self.data = data

    def split_data(self, target_col):
        X = self.data.drop(columns=[target_col, 'TransactionStartTime'])
        y = self.data[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self, X_train, y_train, X_test, y_test):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }

        results = []

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC-AUC': roc_auc
            })

        return pd.DataFrame(results)

    def hyperparameter_tuning(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_, grid_search.best_params_


class ModelTrainingPipeline:
    def __init__(self, data):
        self.data = data
        self.model_training = ModelTraining(data)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.model_training.split_data('RiskLabel')

    def train_models(self):
        self.results = self.model_training.train_models(self.X_train, self.y_train, self.X_test, self.y_test)

    def tune_hyperparameters(self):
        self.best_model, self.best_params = self.model_training.hyperparameter_tuning(self.X_train, self.y_train)

    def get_results(self):
        return self.results, self.best_model, self.best_params


# # Main Workflow
# if __name__ == "__main__":
#     # Sample dataset
#     data = pd.DataFrame({
#         'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
#         'CustomerId': ['C1', 'C2', 'C1', 'C3', 'C3'],
#         'Amount': [100, 200, 150, np.nan, 300],
#         'TransactionStartTime': ['2023-01-01T12:00:00Z', '2023-01-02T14:30:00Z', '2023-01-01T15:45:00Z', '2023-01-03T10:15:00Z', '2023-01-03T11:00:00Z'],
#         'ProductCategory': ['A', 'B', 'A', 'C', 'B'],
#         'FraudResult': [0, 1, 0, 0, 1]
#     })

#     pipeline = CreditScoringPipeline(data)

#     # Execute steps individually
#     pipeline.feature_engineering_steps()
#     pipeline.rfms_and_woe_binning_steps()
#     results, best_model, best_params = pipeline.model_training_steps()

#     # Final Output
#     print("Model Results:")
#     print(results)
#     print("Best Model Parameters:", best_params)
