# machine-learning--estudo_de_caso_sorveteria

Cenário
Imagine que você é proprietário de uma sorveteria chamada Gelato Mágico, localizada em uma cidade litorânea. Você percebe que a quantidade de sorvetes vendidos diariamente tem uma forte correlação com a temperatura ambiente. No entanto, sem um planejamento adequado, você pode acabar produzindo mais sorvetes do que o necessário e ter prejuízos com desperdícios ou, ao contrário, produzir menos e perder vendas.

Para solucionar esse problema, você decide usar Machine Learning para prever quantos sorvetes serão vendidos com base na temperatura. Com esse modelo, será possível antecipar a demanda e planejar a produção de maneira eficiente.

Usei o google colab com seu notebooks para gerar a base de dados via GeminiAI.
# prompt: Crie uma base de dados com o nome de  sorvetes, e como colunas data, vendas e temperatura.  Ele  tem que ter um  total de 100 registros. Devido a variação de temperatura, se nota que há uma forte correlação na venda diaria de sorvetes.  Afim de fazer um planejamento adequado e não produzir mais sorvetes do que necessário, e ter prejuijos ou , ao contrario, produzir menos e perder vendas.

pip install mlflow


import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 100 dates
dates = pd.date_range(start='2023-01-01', periods=100)

# Generate temperatures with some variation
temperatures = np.random.normal(loc=25, scale=5, size=100)
temperatures = np.clip(temperatures, 20, 35)  # Ensure temperatures are within a reasonable range

# Generate sales based on temperature with added noise
sales = 50 + 3 * temperatures + np.random.normal(loc=0, scale=10, size=100)
sales = np.clip(sales, 0, 200)  # Ensure sales are non-negative


# Create the DataFrame
sorvetes_df = pd.DataFrame({'data': dates, 'vendas': sales, 'temperatura': temperatures})

# Display the first few rows of the DataFrame
print(sorvetes_df.head())


# Save the DataFrame to a CSV file (optional)
sorvetes_df.to_csv('sorvetes.csv', index=False)





## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://medium.com/@gilnei809/gilnei-azambuja-borges-analista-de-dados-e-administrador-de-banco-de-dados-8774175b0e46)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](http://www.linkedin.com/in/gilnei-azambuja-borges-1a83432b)
[![KAGGLE](https://img.shields.io/badge/Kaggle-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://www.kaggle.com/gilneiborges)
