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
from src.DataProcessing import (
    load_obj,
    save_obj
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


def fit_rf_model(
        save=True
) -> dict:
    data = pd.read_csv(DATA_PATH)
    x, y = data.drop(columns=['TARGET']), data.TARGET
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('std', StandardScaler(), CONTINUOUS_FEATURES),
            ('ordinal', OrdinalEncoder(), ORDINAL_FEATURES),
        ]
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
        'model__max_depth': [2, 4, 8, 16, 32],
    }
    rf_model = TrainModel(
        x=x,
        y=y,
        test_size=TEST_SIZE
    )
    res = rf_model.train(rf_pipeline_1, rf_params_1, num_rs=5)
    if save:
        save_obj(res, 'D:\\proj\\credit default\\DATA1030_MIDTERM_PROJECT\\rf_model_1_res')
    return res


if __name__ == '__main__':
    fit_rf_model()