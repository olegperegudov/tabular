import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path
import os

# root = os.path.abspath('test_mnist_project\input')
# df_path = os.path.join(root, 'mnist_train.csv')
# new_df_path = os.path.join(root, 'mnist_train_folds.csv')

PROJECT_DIR = Path.cwd().parent

input_path = os.path.join(PROJECT_DIR, 'input')
df_path = os.path.join(input_path, 'mnist_train.csv')
new_df_path = os.path.join(input_path, 'mnist_train_folds.csv')


# def create_folds(n_splits, root, df_path, new_df_path, rewrite=False, shuffle=True):
def create_folds(n_splits, rewrite=False, shuffle=True):
    """Makes a copy of mnist_train.csv with extra column at the end - "kfolds"
       and set fold's number in it.
    Args:
        n_splits (int): number of KFold splits
        root (str): path to root folder
        df_path (str): path to mnist_train.csv
        new_df_path (str): path for new mnist_train_folds.csv
        rewrite (bool, optional): whether to rewrite mnist_train_folds.csv if already exist. Defaults to False.
        shuffle (bool, optional): whether shuffle the df on splits
    """
    # read df
    df = pd.read_csv(df_path)
    # make extra col to fill in later
    # df['kfold'] = None

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=1)

    # loop to set value for kfold column
    for idx, (_, valid) in enumerate(kf.split(df)):
        df.loc[valid, 'kfold'] = idx

    df['kfold'] = df['kfold'].astype('int')

    # check if file exists
    # if doesn't exist - writes new file
    if not os.path.isfile(new_df_path):
        df.to_csv(new_df_path)
    else:
        # if exists but 'rewrite' == True -> rewrites it. Else skips.
        if rewrite:
            df.to_csv(new_df_path)
        else:
            pass


if __name__ == "__main__":
    create_folds(n_splits=5)
