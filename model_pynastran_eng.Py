import pandas as pd
import numpy as np
from pynastran.bdf.bdf import BDF
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class DataProcessor(ABC):
    @abstractmethod
    def load_data(self, file_path):
        pass

    @abstractmethod
    def preprocess_data(self, data):
        pass

class CSVDataProcessor(DataProcessor):
    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def preprocess_data(self, data):
        # Normalização dos dados
        data.fillna(0, inplace=True)
        return data

class FissureCriticalityModel:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = RandomForestRegressor()

    def train_model(self, file_path):
        data = self.data_processor.load_data(file_path)
        preprocessed_data = self.data_processor.preprocess_data(data)

        # Dividir os dados em características (X) e alvo (y)
        X = preprocessed_data.drop(columns=['criticality'])
        y = preprocessed_data['criticality']

        # Dividir os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar o modelo
        self.model.fit(X_train, y_train)

        # Avaliar o modelo
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f'Mean Squared Error: {mse}')

    def identify_criticality(self, file_path):
        data = self.data_processor.load_data(file_path)
        preprocessed_data = self.data_processor.preprocess_data(data)

        # Prever a criticidade usando o modelo treinado
        X = preprocessed_data.drop(columns=['criticality'])
        criticality_scores = self.model.predict(X)

        return np.array(criticality_scores)

# Uso do código
data_processor = CSVDataProcessor()
fissure_model = FissureCriticalityModel(data_processor)
file_path = 'dados_estruturais.csv'

# Treinar o modelo
fissure_model.train_model(file_path)

# Identificar a criticidade
result = fissure_model.identify_criticality(file_path)

print(result)