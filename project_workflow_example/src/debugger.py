from sklearn import tree
from sklearn import metrics
import pandas as pd
import joblib
import os
from pathlib import Path

data_path = os.path.abspath('test_mnist_project\input\mnist_train_folds.csv')

# root = os.path.abspath('test_mnist_project\input')
# df_path = os.path.join(root, 'mnist_train.csv')
# new_df_path = os.path.join(root, 'mnist_train_folds.csv')

df = pd.read_csv(data_path)
print(df.shape)
# print(df.iloc[:10, -3:])
# print(df.kfold.dtype)

# path_1 = Path('input\mnist_train_folds.csv')

# df_path = Path('input/mnist_train.csv')
# df = pd.read_csv(df_path)
# print(df.shape)
# print(path_1)

# print((Path.cwd()).parent)
