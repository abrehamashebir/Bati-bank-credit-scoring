import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class FeatureEngineering:
    def __init__(self, data):
        self.data = data

    def create_aggregate_features(self):
        aggregate_features = self.data.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std']
        })
        aggregate_features.columns = ['total_transaction_amount', 'average_transaction_amount',
                                       'transaction_count', 'std_transaction_amount']
        aggregate_features.reset_index(inplace=True)
        logging.info(aggregate_features)
        return aggregate_features

    def extract_time_features(self):
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'])
        self.data['transaction_hour'] = self.data['TransactionStartTime'].dt.hour
        self.data['transaction_day'] = self.data['TransactionStartTime'].dt.day
        self.data['transaction_month'] = self.data['TransactionStartTime'].dt.month
        self.data['transaction_year'] = self.data['TransactionStartTime'].dt.year
        logging.info('Aggregate Features')
        logging.info(self.data)
        return self.data

    def encode_categorical_variables(self, categorical_columns):
        encoder = LabelEncoder()
        for col in categorical_columns:
            self.data[col] = encoder.fit_transform(self.data[col].astype(str))
        logging.info('Encoding categorical_variables')
        logging.info(self.data)
        return self.data

    def handle_missing_values(self):
        imputer = SimpleImputer(strategy='mean')
        self.data['Amount'] = imputer.fit_transform(self.data[['Amount']])
        logging.info('Handling Missing Values')
        logging.info(self.data)
        return self.data

    def normalize_features(self, numerical_columns):
        scaler = MinMaxScaler()
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])
        logging.info('Normalize Numeric features')
        logging.info(self.data)
        return self.data

class RFMSBinning:
    def __init__(self, data):
        self.data = data

    def calculate_rfms(self):
            # Ensure both datetime objects are timezone-naive
            now = pd.Timestamp.now(tz='UTC').tz_localize(None)  # Make tz-naive
            self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime']).dt.tz_localize(None)# Make tz-naive
            self.data['Recency'] = (now - self.data['TransactionStartTime']).dt.days
            self.data['Frequency'] = self.data.groupby('CustomerId')['TransactionId'].transform('count')
            self.data['Monetary'] = self.data.groupby('CustomerId')['Amount'].transform('sum')
            self.data['Subscription'] = self.data.groupby('CustomerId')['ProductCategory'].transform('nunique')
            return self.data

    def assign_labels(self):
        monetary_threshold = self.data['Monetary'].quantile(0.5)
        frequency_threshold = self.data['Frequency'].quantile(0.5)

        self.data['RiskLabel'] = np.where(
            (self.data['Monetary'] > monetary_threshold) & (self.data['Frequency'] > frequency_threshold), 'Good', 'Bad'
        )
        self.data['RiskLabel'] = self.data['RiskLabel'].map({'Good': 0, 'Bad': 1})
        self.data['Subscription'] = self.data.groupby('CustomerId')['ProductCategory'].transform('nunique')
        logging.info('Labeling Data')
        logging.info(self.data)
        return self.data

    def apply_woe_binning(self):
        woe_transformer = WOE()
        woe_transformer.fit(self.data, self.data['RiskLabel'])
        return woe_transformer.transform(self.data)



class CreditScoringPipeline:
    def __init__(self, data):
        self.data = data
        self.feature_engineering = FeatureEngineering(data)
        self.rfms_binning = RFMSBinning(data)
        # self.model_training = ModelTraining(data)

    def feature_engineering_steps(self):
        self.data = self.feature_engineering.create_aggregate_features()
        self.data = self.feature_engineering.extract_time_features()
        self.data = self.feature_engineering.encode_categorical_variables(['ProductCategory'])
        self.data = self.feature_engineering.handle_missing_values()
        self.data = self.feature_engineering.normalize_features(['Amount'])

    def rfms_and_woe_binning_steps(self):
        self.data = self.rfms_binning.calculate_rfms()
        self.data = self.rfms_binning.assign_labels()
        self.data = self.rfms_binning.apply_woe_binning()

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
