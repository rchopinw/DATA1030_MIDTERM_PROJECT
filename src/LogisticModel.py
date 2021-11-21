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


def fit_lr_model(
        save=True
) -> dict:
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
    dimension_reduction = ColumnTransformer(
        transformers=[
            ('dim_reduction', TruncatedSVD(n_components=120, random_state=RANDOM_SEED), determine.determine_sparse)
        ]
    )
    # random forest model
    lr_pipeline_1 = Pipeline(
        steps=[
            ('preprocess', preprocessor),
            ('dim_reduction', dimension_reduction),
            ('sampling', SMOTE(random_state=RANDOM_SEED)),
            ('model', LogisticRegression(max_iter=20000)
            )
        ]
    )
    lr_params_1 = {"model__C": 1 / np.logspace(-2, 2, 10)}
    lr_model = TrainModel(
        x=x,
        y=y,
        test_size=TEST_SIZE
    )
    res = lr_model.train(lr_pipeline_1, lr_params_1, num_rs=5)
    if save:
        save_obj(res, 'D:\\proj\\credit default\\DATA1030_MIDTERM_PROJECT\\lr_model_1_res')
    return res


if __name__ == '__main__':
    fit_lr_model()