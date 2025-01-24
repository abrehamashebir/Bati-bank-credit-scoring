
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from abc import ABC, abstractmethod
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Define Interfaces
class DataLoader(ABC):
    @abstractmethod
    def load(self, file_path):
        pass

class DataSummarizer(ABC):
    @abstractmethod
    def summarize(self, df):
        pass

class NumericalFeatureVisualizer(ABC):
    @abstractmethod
    def visualize(self, df, numerical_cols):
        pass

class CategoricalFeatureVisualizer(ABC):
    @abstractmethod
    def visualize(self, df, categorical_cols):
        pass

class CorrelationAnalyzer(ABC):
    @abstractmethod
    def analyze(self, df, numerical_cols):
        pass

class MissingValueIdentifier(ABC):
    @abstractmethod
    def identify(self, df):
        pass

class OutlierDetector(ABC):
    @abstractmethod
    def detect(self, df, numerical_cols):
        pass

# 2. Concrete Implementations
class PandasDataLoader(DataLoader):
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

class PandasSummarizer(DataSummarizer):
    def summarize(self, df):
        logging.info('Performing summary statistics')
        logging.info(f'Number of rows: {df.shape[0]}')
        logging.info(f'Number of columns: {df.shape[1]}')
        logging.info(f'Data types: \n{df.dtypes}')
        logging.info(f"First 5 rows of the DataFrame:\n{df.head()}")
        logging.info(f"Summary statistics of numerical features:\n{df.describe()}")

class SeabornHistPlotter(NumericalFeatureVisualizer):
    def visualize(self, df, numerical_cols):
        logging.info('Visualizing numerical features distributions')
        for col in numerical_cols:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

class SeabornCountPlotter(CategoricalFeatureVisualizer):
    def visualize(self, df, categorical_cols):
        logging.info("Visualizing categorical feature distributions")
        for col in categorical_cols:
            plt.figure(figsize=(8,5))
            sns.countplot(x=df[col])
            plt.title(f'Distribution of {col}')
            plt.show()

class PandasCorrelationAnalyzer(CorrelationAnalyzer):
    def analyze(self, df, numerical_cols):
         logging.info("Performing correlation analysis")
         plt.figure(figsize=(10, 8))
         sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
         plt.title('Correlation Matrix')
         plt.show()

class PandasMissingValueIdentifier(MissingValueIdentifier):
    def identify(self, df):
        logging.info('Identifying missing values')
        logging.info(f'Missing Values:\n{df.isnull().sum()}')

class SeabornOutlierDetector(OutlierDetector):
    def detect(self, df, numerical_cols):
       logging.info('Detecting outliers')
       for col in numerical_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df[col])
            plt.title(f'Box Plot of {col}')
            plt.show()

# 3. Pipeline Class
class EDAPipeline:
    def __init__(self, data_loader, data_summarizer, numerical_visualizer, categorical_visualizer, correlation_analyzer, missing_identifier, outlier_detector):
        self.data_loader = data_loader
        self.data_summarizer = data_summarizer
        self.numerical_visualizer = numerical_visualizer
        self.categorical_visualizer = categorical_visualizer
        self.correlation_analyzer = correlation_analyzer
        self.missing_identifier = missing_identifier
        self.outlier_detector = outlier_detector

    def run(self, file_path, load_only=False):
        logging.info('Starting EDA pipeline')
        df = self.data_loader.load(file_path)
        if df is None:
           return

        self.data_summarizer.summarize(df)
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        self.numerical_visualizer.visualize(df, numerical_cols)
        self.categorical_visualizer.visualize(df, categorical_cols)
        self.correlation_analyzer.analyze(df, numerical_cols)
        self.missing_identifier.identify(df)
        self.outlier_detector.detect(df, numerical_cols)
        logging.info('EDA pipeline completed successfully.')

# if __name__ == '__main__':
#     #Get file path from environment variables, defaults to cs-training.csv
#     file_path = os.environ.get('DATA_FILE_PATH', 'cs-training.csv')
#     #Initialize components
#     data_loader = PandasDataLoader()
#     data_summarizer = PandasSummarizer()
#     numerical_visualizer = SeabornHistPlotter()
#     categorical_visualizer = SeabornCountPlotter()
#     correlation_analyzer = PandasCorrelationAnalyzer()
#     missing_identifier = PandasMissingValueIdentifier()
#     outlier_detector = SeabornOutlierDetector()

#     #Run the pipeline
#     pipeline = EDAPipeline(data_loader, data_summarizer, numerical_visualizer, categorical_visualizer, correlation_analyzer, missing_identifier, outlier_detector)
#     pipeline.run(file_path)
