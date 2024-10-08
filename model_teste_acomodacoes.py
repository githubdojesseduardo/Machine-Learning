import pandas as pd
import numpy as np
from scipy import stats

# Dados de clientes e suas preferências de hospedagem
data = {
    'cliente_id': range(1, 101),
    'idade': np.random.randint(18, 70, 100),
    'renda': np.random.randint(2000, 10000, 100),
    'preferencia': np.random.choice(['casa', 'apartamento', 'chalé'], 100)
}

# Criar DataFrame
df = pd.DataFrame(data)

# Função para realizar o teste A/B
def ab_test(dataframe, group_col, target_col):
    # Contagem de preferências por grupo
    group_counts = dataframe.groupby(group_col)[target_col].value_counts().unstack().fillna(0)
    
    # Calcular proporções
    group_props = group_counts.div(group_counts.sum(axis=1), axis=0)
    
    # Teste qui-quadrado
    chi2, p, _, _ = stats.chi2_contingency(group_counts)
    
    return group_counts, group_props, chi2, p

# Realizar o teste A/B
group_counts, group_props, chi2, p = ab_test(df, 'preferencia', 'cliente_id')

# Preparar dados para exibição no Power BI
df_resultados = group_counts.reset_index()
df_resultados.columns.name = None

# Salvar os dados tratados em um arquivo CSV
df_resultados.to_csv('resultados_ab_test.csv', index=False)

# Exibir resultados
print("Contagem de preferências por grupo:")
print(group_counts)
print("\nProporções de preferências por grupo:")
print(group_props)
print(f"\nValor do qui-quadrado: {chi2}")
print(f"Valor p: {p}")

# Interpretação dos resultados
if p < 0.05:
    print("\nHá uma diferença estatisticamente significativa nas preferências dos clientes.")
else:
    print("\nNão há uma diferença estatisticamente significativa nas preferências dos clientes.")

