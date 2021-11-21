# Importing packages
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD, DictionaryLearning, NMF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from src.framework import TrainModel, DetermineSparse
from src.constants import (
    DATA_PATH,
    TEST_SIZE,
    RANDOM_SEED,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    ORDINAL_FEATURES
)



def fit_boosting_model(
        dimension_reduction,
        balance_sampling,
) -> tuple:
    pass


def fit_knn_model(
        dimension_reduction,
        balance_sampling,
) -> tuple:
    pass


if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    x, y = data.drop(columns=['TARGET']), data.TARGET
    determine = DetermineSparse()
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('std', StandardScaler(), CONTINUOUS_FEATURES),
            ('ordinal', OrdinalEncoder(), ORDINAL_FEATURES),
        ]
    )

    # svc model
    svc_models, svc_scores = fit_svc_model(
        dimension_reduction=TruncatedSVD(
            n_components=30,
            random_state=RANDOM_SEED,
        ),
        balance_sampling=SMOTE(
            random_state=RANDOM_SEED
        )
    )

    # random forest model
    rf_pipeline_1 = Pipeline(
        steps=[
            ('preprocess', preprocessor),
            ('sampling', SMOTE(random_state=RANDOM_SEED)),
            ('model', RandomForestClassifier(random_state=RANDOM_SEED))
        ]
    )
    rf_params_1 = {
        'model__n_estimators': [10, 50, 100, 150, 200, 300, 500],
        'model__max_depth': [2, 4, 8, 16, 32]
    }
    rf_models, rf_scores = fit_random_forest_model(
        rf_pipeline_1,
        rf_params_1
    )