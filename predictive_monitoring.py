import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Carregar dados
data = pd.read_csv('dados_alertas.csv')

# Pré-processamento
data['alert_level'] = data['alert_type'].map({'Critical': 3, 'Major': 2, 'Warning': 1, 'OK': 0})

# Divisão dos dados
X = data[['alert_level', 'other_features']]
y = data['remaining_life']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

print(f'MAE: {mae}, MSE: {mse}, R²: {r2}')
