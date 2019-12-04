import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path
import codecs
import category_encoders as ce


class MovielensLoader:
    def __init__(self, data_dir=Path("./ml-100k"), user_filename="u.user", item_filename="u.item"):
        self.data_dir = data_dir
        self.user_path = data_dir / user_filename
        self.item_path = data_dir / item_filename

    def create_dataset(self, include_user_features=True, include_item_features=True):
        df_train, y_train = self.load_log_and_ratings(log_filename="ua.base")
        df_test, y_test = self.load_log_and_ratings(log_filename="ua.test")
        target_col = ["uid", "mid"]

        if include_user_features:
            users = self.load_users()
            target_col = list(set(target_col + users.columns.tolist()))
            df_train = pd.merge(df_train, users, on="uid")
            df_test = pd.merge(df_test, users, on="uid")

        if include_item_features:
            items = self.load_items()
            df_train = pd.merge(df_train, items, on="mid")
            df_test = pd.merge(df_test, items, on="mid")

        self.encoder = ce.OneHotEncoder(cols=target_col)

        X_train = self.encoder.fit_transform(df_train)
        X_test = self.encoder.transform(df_test)

        X_train = csr_matrix(X_train, dtype=np.float)
        X_test = csr_matrix(X_test, dtype=np.float)
        return X_train, y_train, X_test, y_test

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

    def load_items(self):
        with codecs.open(self.item_path, "r", "Shift-JIS", "ignore") as f:
            items = pd.read_table(f, names=["mid", "title", "released", "video_released", "IMDb_URL",
                                            "unknown", "Action", "Adventure", "Animation",
                                            "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                                                       "Film_Noir", "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
                                                       "Thriller", "War", "Western"], delimiter="|", dtype=str)
        items = items.drop(["title", "released", "video_released", "IMDb_URL"], axis=1)
        return items
