import pandas as pd
import numpy as np
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from sklearn.preprocessing import StandardScaler

# Carregar dados
data = pd.read_csv('transacoes_financeiras.csv')

# Selecionar características relevantes
features = ['valor_transacao', 'hora_transacao', 'local_transacao']
X = data[features]

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Detectar anomalias com Floresta de Isolamento
clf_iforest = IForest(contamination=0.05)
clf_iforest.fit(X_scaled)
data['anomaly_iforest'] = clf_iforest.predict(X_scaled)

# Detectar anomalias com LOF
clf_lof = LOF(contamination=0.05)
clf_lof.fit(X_scaled)
data['anomaly_lof'] = clf_lof.predict(X_scaled)

# Filtrar transações anômalas
anomalies_iforest = data[data['anomaly_iforest'] == 1]
anomalies_lof = data[data['anomaly_lof'] == 1]

# Exibir resultados
print("Anomalias detectadas pela Floresta de Isolamento:")
print(anomalies_iforest)
print("\nAnomalias detectadas pelo LOF:")
print(anomalies_lof)
