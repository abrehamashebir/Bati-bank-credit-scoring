
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from abc import ABC, abstractmethod
import os


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Interface Definitions
class DataLoaderInterface(ABC):
    @abstractmethod
    def load(self, file_path):
        pass

class DataSummarizerInterface(ABC):
    @abstractmethod
    def summarize(self, df):
        pass

class NumericalFeatureVisualizerInterface(ABC):
    @abstractmethod
    def visualize(self, df, numerical_cols):
        pass

class CategoricalFeatureVisualizerInterface(ABC):
    @abstractmethod
    def visualize(self, df, categorical_cols):
        pass

class CorrelationAnalyzerInterface(ABC):
    @abstractmethod
    def analyze(self, df, numerical_cols):
        pass

class MissingValueIdentifierInterface(ABC):
    @abstractmethod
    def identify(self, df):
        pass

class OutlierDetectorInterface(ABC):
    @abstractmethod
    def detect(self, df, numerical_cols):
        pass

# Concrete Implementations
class DataLoader(DataLoaderInterface):
    def load(self, file_path):
        try:
            logging.info(f'Loading data from: {file_path}')
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                logging.error(f"Unsupported file type: {file_path}")
                return None

            if df is not None:
                unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed')]
                if unnamed_cols:
                    df.drop(columns=unnamed_cols, inplace=True)
                    logging.info(f'Data loaded successfully with shape: {df.shape}')
            return df
        except Exception as e:
            logging.error(f'Error loading data: {e}')
            return None

class DataSummarizer(DataSummarizerInterface):
    def summarize(self, df):
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        logging.info('Performing summary statistics')
        logging.info(f'Number of rows: {df.shape[0]}')
        logging.info(f'Number of columns: {df.shape[1]}')
        logging.info(f'Data types: \n{df.dtypes}')
        logging.info(f"First 5 rows of the DataFrame:\n{df.head()}")
        logging.info(f"Summary statistics of numerical features:\n{df.describe()}")

class NumericalFeatureVisualizer(NumericalFeatureVisualizerInterface):
    def visualize(self, df, numerical_cols):
        logging.info('Visualizing numerical feature distributions')
        for col in numerical_cols:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

class CategoricalFeatureVisualizer(CategoricalFeatureVisualizerInterface):
    def visualize(self, df, categorical_cols):
        logging.info('Visualizing categorical feature distributions')
        for col in categorical_cols:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=df[col])
            plt.title(f'Distribution of {col}')
            plt.show()

class CorrelationAnalyzer(CorrelationAnalyzerInterface):
    def analyze(self, df, numerical_cols):
        logging.info('Performing correlation analysis')
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

class MissingValueIdentifier(MissingValueIdentifierInterface):
    def identify(self, df):
        total_rows = len(df)
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / total_rows) * 100
        missing_summary = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage (%)': missing_percentage
        }).sort_values(by='Missing Percentage (%)', ascending=False)
    
        # Filter for columns with missing values
        missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
        logging.info('Identifying missing values')
        logging.info(f'Missing Values:\n{missing_values}')
        logging.info(f'Percentage of Missing Values:\n{missing_percentage} %')

class OutlierDetector(OutlierDetectorInterface):
    def detect(self, df, numerical_cols):
        logging.info('Detecting outliers')
        for col in numerical_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df[col])
            plt.title(f'Box Plot of {col}')
            plt.show()

# Updated EDAPipeline Class
class EDAPipeline:
    def __init__(self):
        self.loader = DataLoader()
        self.summarizer = DataSummarizer()
        self.num_visualizer = NumericalFeatureVisualizer()
        self.cat_visualizer = CategoricalFeatureVisualizer()
        self.corr_analyzer = CorrelationAnalyzer()
        self.missing_identifier = MissingValueIdentifier()
        self.outlier_detector = OutlierDetector()

    def load_data(self, file_path):
        return self.loader.load(file_path)

    def summarize_data(self, df):
        self.summarizer.summarize(df)

    def visualize_numerical_features(self, df):
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.num_visualizer.visualize(df, numerical_cols)

    def visualize_categorical_features(self, df):
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        self.cat_visualizer.visualize(df, categorical_cols)

    def analyze_correlation(self, df):
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.corr_analyzer.analyze(df, numerical_cols)

    def identify_missing_values(self, df):
        self.missing_identifier.identify(df)

    def detect_outliers(self, df):
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.outlier_detector.detect(df, numerical_cols)
# if __name__ == '__main__':
#     # Initialize pipeline
#     pipeline = EDAPipeline()

#     # File path setup
#     file_path = os.environ.get('DATA_FILE_PATH', 'cs-training.csv')

#     # Load data
#     df_loaded = pipeline.load_data(file_path)

#     if df_loaded is not None:
#         print(f"Data loaded successfully with shape {df_loaded.shape}")

#         # Run each step
#         pipeline.summarize_data(df_loaded)
#         pipeline.visualize_numerical_features(df_loaded)
#         pipeline.visualize_categorical_features(df_loaded)
#         pipeline.analyze_correlation(df_loaded)
#         pipeline.identify_missing_values(df_loaded)
#         pipeline.detect_outliers(df_loaded)
#     else:
#         print("Data loading failed")

