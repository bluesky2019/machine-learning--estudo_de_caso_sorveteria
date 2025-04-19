
# criei esse codigo para gerar o grafico para representar o meu o resultado depois de executado o codigo de machine learning: "Relação entre Temperatura e Vendas de Sorvetes"
import matplotlib.pyplot as plt

# Load the data (assuming 'sorvetes.csv' is in the same directory)
sorvetes_df = pd.read_csv('/content/sorvetes.csv')

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(sorvetes_df['temperatura'], sorvetes_df['vendas'], color='blue', label='Vendas de sorvetes')
plt.xlabel('Temperatura')
plt.ylabel('Vendas')
plt.title('Relação entre Temperatura e Vendas de Sorvetes')
plt.legend()
plt.grid(True)
plt.show()
