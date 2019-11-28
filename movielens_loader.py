import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer


class MovielensLoader:
    def __init__(self, data_dir=Path("./ml-100k"), user_filename="u.user"):
        self.data_dir = data_dir
        self.user_path = data_dir / user_filename

    def create_dataset(self, include_context_features=True):
        """datasetを作成.
        :param include_context_features: context featureをXに含めるか
        """
        users = self.load_users()
        df_train, y_train = self.load_log_and_ratings(log_filename="ua.base")
        df_test, y_test = self.load_log_and_ratings(log_filename="ua.test")
        if include_context_features:
            df_train = self.merge(df_train, users)
            df_test = self.merge(df_test, users)
        train_data = self.to_dict(df_train)
        test_data = self.to_dict(df_test)

        self.vectorizer = DictVectorizer()
        X_train = self.vectorizer.fit_transform(train_data)
        X_test = self.vectorizer.transform(test_data)
        X_train = csr_matrix(X_train, dtype=np.float)
        X_test = csr_matrix(X_test, dtype=np.float)
        return X_train, y_train, X_test, y_test

    def to_dict(self, df):
        return df.to_dict(orient='records')

    def load_log_and_ratings(self, log_filename, drop_columns=["timestamp"]):
        logs = pd.read_csv(self.data_dir / log_filename, names=["uid", "mid", "rating", "timestamp"], sep="\t", dtype=str)
        ratings = np.array(logs["rating"], dtype=np.float)
        drop_columns.append("rating")
        logs = logs.drop(drop_columns, axis=1)
        return logs, ratings

    def load_users(self, drop_columns=["age", "zip_code"]):
        users = pd.read_csv(self.user_path, names=["uid", "age", "gender", "occupation", "zip_code"], sep="|", dtype=str)
        users = users.drop(drop_columns, axis=1)
        return users

    def merge(self, logs, users):
        return pd.merge(logs, users, on="uid")
