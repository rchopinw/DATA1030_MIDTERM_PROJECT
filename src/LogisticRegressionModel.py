# Importing packages
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.TrainModelCommon import TrainModel
from src.DataProcessing import (
    save_obj,
    DetermineSparse
)
from src.constants import (
    DI1_FEATURES,
    DI2_FEATURES,
    RANDOM_SEED
)


def logistic_regression_full_loop():
    for ordinal, continuous, categorical, data_path in [
        x.values()
        for x in [
            DI1_FEATURES, DI2_FEATURES
        ]
    ]:
        file = data_path.split('.')[0].split('/')[-1]
        data = pd.read_csv(data_path)
        determine = DetermineSparse()
        x, y = data.drop(columns=['TARGET']), data.TARGET
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical),
                ('std', StandardScaler(), continuous),
                ('ordinal', OrdinalEncoder(), ordinal),
            ]
        )
        dimension_reduction = ColumnTransformer(
            transformers=[
                ('dim_reduction', TruncatedSVD(n_components=60, random_state=RANDOM_SEED), determine.determine_sparse)
            ]
        )
        # random forest model
        lr_pipeline = Pipeline(
            steps=[
                ('preprocess', preprocessor),
                ('dim_reduction', dimension_reduction),
                ('model', LogisticRegression(max_iter=20000)
                 )
            ]
        )
        lr_params = {"model__C": 1 / np.logspace(-2, 2, 10)}
        lr_model = TrainModel(
            x=x,
            y=y,
            test_size=0.1
        )
        res = lr_model.train(lr_pipeline, lr_params, num_rs=5, set_weight=False)
        save_obj(res, 'data/logistic_regression_model_{}'.format(file))
