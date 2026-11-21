
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["tenure_group"] = pd.cut(
            X["Tenure"],
            bins=[-1, 2, 5, 10],
            labels=["New", "Medium", "Long"]
        )

        X["balance_to_salary_ratio"] = (
            X["Balance"] / X["EstimatedSalary"].replace(0, np.nan)
        )

        X["products_per_tenure"] = (
            X["NumOfProducts"] / X["Tenure"].replace(0, 1)
        )

        return X


class IQRCapper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1

        self.lower_bounds_ = q1 - 1.5 * iqr
        self.upper_bounds_ = q3 + 1.5 * iqr

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        return np.clip(
            X,
            self.lower_bounds_,
            self.upper_bounds_
        )
