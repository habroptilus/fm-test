import numpy as np
import pandas as pd
import codecs
from scipy.sparse import csr_matrix
from pathlib import Path


class MovielensLoader:
    def __init__(self, data_dir=Path("./ml-100k"), log_filename="u.data", item_filename="u.item", user_filename="u.user"):
        self.log_path = data_dir / log_filename
        self.item_path = data_dir / item_filename
        self.user_path = data_dir / user_filename

    def create_dataset(self):
        items = self.load_items()
        users = self.load_users()
        logs, y = self.load_log_and_ratings()
        features_df = self.merge(logs, users, items)
        X = csr_matrix(features_df.values, dtype=np.float)
        return X, y

    def load_log_and_ratings(self, drop_columns=["timestamp"]):
        logs = pd.read_csv(self.log_path, names=["uid", "mid", "rating", "timestamp"], sep="\t")
        ratings = np.array(logs["rating"])
        drop_columns.append("rating")
        logs = logs.drop(drop_columns, axis=1)
        return logs, ratings

    def load_users(self, drop_columns=["age", "zip_code"]):
        users = pd.read_csv(self.user_path, names=["uid", "age", "gender", "occupation", "zip_code"], sep="|")
        users = users.drop(drop_columns, axis=1)
        return users

    def load_items(self, drop_columns=["title", "released", "video_released", "IMDb_URL"]):
        """itemのcontext追加情報の読み込み
        read_csvだとUnicodeDecodeErrorが起きてしまったのでhttps://qiita.com/niwaringo/items/d2a30e04e08da8eaa643で解決
        後ろのcolumnはジャンルを表す.(複数ジャンルに属するitemもあるらしい)
        """

        with codecs.open(self.item_path, "r", "Shift-JIS", "ignore") as f:
            items = pd.read_table(f, names=["mid", "title", "released", "video_released", "IMDb_URL",
                                            "unknown", "Action", "Adventure", "Animation",
                                            "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                                            "Film_Noir", "Horror", "Musical", "Mystery", "Romance", "Sci_Fi",
                                            "Thriller", "War", "Western"], delimiter="|")
        items = items.drop(drop_columns, axis=1)
        return items

    def merge(self, logs, users, items):
        log_user = pd.merge(logs, users, on="uid")
        log_user_dummied = pd.get_dummies(log_user)
        merged_df = pd.merge(log_user_dummied, items, on="mid")
        return pd.get_dummies(merged_df, columns=["uid", "mid"])
