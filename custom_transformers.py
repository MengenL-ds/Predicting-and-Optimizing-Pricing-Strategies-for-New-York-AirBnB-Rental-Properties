import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class HostSinceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        host_since = pd.to_datetime(X['host_since'], errors='coerce')
        today = pd.to_datetime("today")

        transformed = pd.DataFrame({
            'host_year': host_since.dt.year,
            'host_month': host_since.dt.month,
            'host_days_active': (today - host_since).dt.days
        })

        return transformed

    def get_feature_names_out(self, input_features=None):
        return ['host_year', 'host_month', 'host_days_active']
    

# Make new binary features based on host_location
class UKHostBinaryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X['host_location'].fillna("").copy()
        return pd.DataFrame({
            'is_uk_host': X_.str.contains("United Kingdom", na=False).astype(int),
            'outside_uk_host': (~X_.str.contains("United Kingdom", na=False)).astype(int)
        }, index=X.index)
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['is_uk_host', 'outside_uk_host'])
    
class BathroomExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, fillna=None):
        self.fillna = fillna  # Allow user-defined default (e.g., 1.0)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        extracted = X['bathrooms_text'].str.extract(r'(\d+\.?\d*)')[0].astype(float)
        if self.fillna is not None:
            extracted = extracted.fillna(self.fillna)
        return extracted.to_frame(name='bathrooms_extracted')

    def get_feature_names_out(self, input_features=None):
        return np.array(['bathrooms_extracted'])
    
class HostResponseOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.response_order = [
            'a few days or more',
            'within a day',
            'within a few hours',
            'within an hour',
            'Unknown'
        ]
        self.encoder = OrdinalEncoder(categories=[self.response_order], handle_unknown='use_encoded_value', unknown_value=len(self.response_order))
        
    def fit(self, X, y=None):
        X_ = X['host_response_time'].fillna('Unknown').copy()
        X_[~X_.isin(self.response_order)] = 'Unknown'
        return self.encoder.fit(X_.to_frame())

    def transform(self, X):
        X_ = X['host_response_time'].fillna('Unknown').copy()
        X_[~X_.isin(self.response_order)] = 'Unknown'
        return pd.DataFrame(self.encoder.transform(X_.to_frame()), columns=['host_response_time_encoded'], index=X.index)
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['host_response_time_encoded'])