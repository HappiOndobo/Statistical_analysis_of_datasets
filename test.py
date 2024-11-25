import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
import seaborn as sns
iris_dataset = load_iris()
iris = datasets.load_iris()
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Названия цветов: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
# Uploading the iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd. DataFrame(data=iris.data, columns=iris.feature_names)
# Add the target variable to the dataframe
df['target'] = iris.target
# Creating a Correlation Matrix
corr_matrix = df.corr()
# Creating a correlation diagram
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Iris dataset correlation diagram')
plt.show()
