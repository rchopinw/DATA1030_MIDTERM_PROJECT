# Importing packages
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier
)
from src.TrainModelCommon import TrainModel
from src.DataProcessing import (
    save_obj,
)
from src.constants import (
    DI1_FEATURES,
    DI2_FEATURES
)


def random_forest_full_loop():
    for ordinal, continuous, categorical, data_path in [
        x.values()
        for x in [
            DI1_FEATURES, DI2_FEATURES
        ]
    ]:
        file = data_path.split('.')[0].split('/')[-1]
        data = pd.read_csv(data_path)
        data = data.dropna()
        x, y = data.drop(columns=['TARGET']), data.TARGET
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical),
                ('std', StandardScaler(), continuous),
                ('ordinal', OrdinalEncoder(), ordinal),
            ]
        )

        # random forest model
        rf_pipeline = Pipeline(
            steps=[
                ('preprocess', preprocessor),
                ('model', RandomForestClassifier())
            ]
        )
        rf_params = {
            'model__n_estimators': [100, 150, 300, 500, 1000],
            'model__max_depth': [2, 4, 8, 16],
        }
        rf_model = TrainModel(
            x=x,
            y=y,
            test_size=0.1
        )
        res = rf_model.train(rf_pipeline, rf_params, num_rs=5)
        save_obj(res, 'data/random_forest_model_{}'.format(file))
