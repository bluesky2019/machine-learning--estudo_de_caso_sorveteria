import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings("ignore")

# ##
# 1. Gerando os dados
# ##
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100)
temperatures = np.random.normal(loc=25, scale=5, size=100)
temperatures = np.clip(temperatures, 20, 35)

# Venda como variável contínua (não binária)
sales = 50 + 3 * temperatures + np.random.normal(loc=0, scale=10, size=100)
sales = np.clip(sales, 0, 200)

# Criar DataFrame
df = pd.DataFrame({
    'data': dates,
    'temperatura': temperatures,
    'vendas': sales
})

# Salvar CSV (opcional)
df.to_csv('sorvetes.csv', index=False)

# ##
# 2. Carregando e preparando os dados
# ##
df = pd.read_csv('sorvetes.csv')
df['data'] = pd.to_datetime(df['data'])

# Features temporais
df['mes'] = df['data'].dt.month
df['dia_semana'] = df['data'].dt.dayofweek
df['fim_de_semana'] = df['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)

# Separar treino/teste (simulando previsão futura)
train = df.iloc[:-10]
future = df.iloc[-10:]

X_train = train[['temperatura', 'mes', 'dia_semana', 'fim_de_semana']]
y_train = train['vendas']
X_test = future[['temperatura', 'mes', 'dia_semana', 'fim_de_semana']]
y_test = future['vendas']

# Padronização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ##
# 3. Treinamento e avaliação 
# ##
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ##
# 4. Log com MLflow
# ##
with mlflow.start_run():
    mlflow.log_param("modelo", "Linear Regression")
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    mlflow.sklearn.log_model(model, "regression_model")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"MLflow run URL: {mlflow.get_artifact_uri()}")

# ##
# 5. Comparar previsões reais vs previstas
# ##
print("\n Previsões para os últimos 10 dias:")
resultados = future[['data']].copy()
resultados['vendas_reais'] = y_test.values
resultados['vendas_previstas'] = y_pred.round(2)
print(resultados)
