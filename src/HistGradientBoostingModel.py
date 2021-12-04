import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier
)
from src.TrainModelCommon import TrainModel
from src.DataProcessing import (
    save_obj,
)
from src.constants import (
    DI1_FEATURES,
    DI2_FEATURES
)


def hist_gradient_boosting_full_loop():
    for ordinal, continuous, categorical, data_path in [
        x.values()
        for x in [
            DI1_FEATURES, DI2_FEATURES
        ]
    ]:
        file = data_path.split('.')[0].split('/')[-1]
        data = pd.read_csv(data_path)
        x, y = data.drop(columns=['TARGET']), data.TARGET
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical),
                ('std', StandardScaler(), continuous),
                ('ordinal', OrdinalEncoder(), ordinal),
            ]
        )
        hgb_pipeline = Pipeline(
            steps=[
                ('preprocess', preprocessor),
                ('model', HistGradientBoostingClassifier())
            ]
        )
        hgb_params = {
            'model__max_depth': [2, 4, 8, 16, 32],
            'model__max_leaf_nodes': [4, 8, 16, 32, 64, 128]
        }
        hgb_model = TrainModel(
            x=x,
            y=y,
            test_size=0.1
        )
        res = hgb_model.train(hgb_pipeline, hgb_params, num_rs=5, set_weight=False)
        save_obj(res, 'data/hist_gradient_boosting_model_{}'.format(file))

