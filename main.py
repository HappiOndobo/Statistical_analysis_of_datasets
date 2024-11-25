# Load the Iris dataset
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
data = sns.load_dataset('iris')
print(data.head())

# Create Pairwise Scatter Plots
sns.pairplot(data, hue='species', diag_kind='hist')
plt.savefig("pairwise_scatter_plot.png")
plt.show()

# Plot Class Frequency Histogram
data['species'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Frequency of Each Class')
plt.savefig("class_frequency_histogram.png")

# Create Correlation Heatmap
correlation_matrix = data.iloc[:, :-1].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.savefig("correlation_heatmap.png")
