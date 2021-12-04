from sklearn.model_selection import (
    ParameterGrid,
    train_test_split
)
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix
)
from src.DataProcessing import (
    FeatureSelector,
    CategoricalFeatureTransformer,
    save_obj
)
from src.constants import (
    DI1_FEATURES,
    DI2_FEATURES,
    DU1_FEATURES,
    DU2_FEATURES
)
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler
)


def train_xgboost_model_weighted(
        x, y,
        preprocess,
        params,
        metric,
        path,
        test_metric=confusion_matrix,
        random_states=None,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        save=True
) -> dict:
    if random_states is None:
        random_states = [x*42 for x in range(1, 5)]

    params_grid = ParameterGrid(params)
    overall_scores = {}

    for rs in random_states:
        print('Executing random state {}.'.format(rs))
        cur_val_scores = []
        cur_train_scores = []
        cur_important_features = []
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x, y,
            test_size=test_size,
            random_state=rs,
            stratify=y
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val,
            train_size=train_size / (train_size + val_size),
            random_state=rs,
            stratify=y_train_val
        )
        x_train = preprocess.fit_transform(x_train)
        x_val = preprocess.transform(x_val)

        # creating 3 separate models for feature selection, hyper-parameter tuning and optimal model production
        xgb_model = xgb.XGBClassifier(
            gpu_id=0,
            n_jobs=-1,
            seed=rs,
            learning_rate=0.1,
            tree_method='gpu_hist',
            scale_pos_weight=sum(1 - y_train_val) / sum(y_train_val)
        )
        xgb_model_optimal = xgb.XGBClassifier(
            gpu_id=0,
            n_jobs=-1,
            seed=rs,
            learning_rate=0.1,
            tree_method='gpu_hist',
            scale_pos_weight=sum(1 - y_train_val) / sum(y_train_val)
        )

        for i in range(len(params_grid)):
            # fitting the xgb model for feature selection
            fs_map = xgb.XGBClassifier(
                gpu_id=0,
                n_jobs=-1,
                seed=rs,
                learning_rate=0.1,
                tree_method='gpu_hist',
                scale_pos_weight=sum(1 - y_train_val) / sum(y_train_val),
                **params_grid[i]
            ).fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=False
            ).feature_importances_ > 0

            # fetch the optimal features by feature importance
            imp_idx = np.array(
                [x for x in range(x_train.shape[1])]
            )[fs_map]

            # using the selected feature index to continue training
            xgb_model.set_params(**params_grid[i])
            xgb_model.fit(
                x_train[:, imp_idx],
                y_train,
                eval_set=[(x_val[:, imp_idx], y_val)],
                verbose=False
            )

            # predicting on validation set and training set
            y_val_pred = xgb_model.predict_proba(
                x_val[:, imp_idx]
            )
            y_train_pred = xgb_model.predict_proba(
                x_train[:, imp_idx]
            )

            # evaluating on validation and training sets (ROC_AUC score is applied)
            cur_val_scores.append(
                metric(y_val, y_val_pred[:, 1])
            )
            cur_train_scores.append(
                metric(y_train, y_train_pred[:, 1])
            )
            cur_important_features.append(imp_idx)

        # find the optimal parameter
        optimal_idx = np.argmax(cur_val_scores)
        optimal_params = list(params_grid)[optimal_idx]
        optimal_features = cur_important_features[optimal_idx]

        xgb_model_optimal.set_params(**optimal_params)

        # re-train on full data (x_train and x_val)
        x_train_val = preprocess.fit_transform(
            x_train_val
        )
        x_test = preprocess.transform(
            x_test
        )

        xgb_model_optimal.fit(
            x_train_val[:, optimal_features],
            y_train_val,
            eval_set=[(x_val[:, optimal_features], y_val)],
            verbose=False
        )
        y_test_pred = xgb_model_optimal.predict(
            x_test[:, optimal_features]
        )
        model_pipeline = Pipeline(
            steps=[
                   ('preprocess', preprocess),
                   ('feature_selection', FeatureSelector(columns=optimal_features)),
                   ('model', xgb_model_optimal)
            ]
        )
        overall_scores[rs] = {
            'test_score': test_metric(y_test, y_test_pred),
            'val_scores': cur_val_scores,
            'train_scores': cur_train_scores,
            'optimal_params': optimal_params,
            'optimal_features': optimal_features,
            'random_state': rs,
            'y_test_pred': y_test_pred,
            'optimal_model_object': model_pipeline
        }
    if save:
        save_obj(
            overall_scores,
            name=path
        )
    return overall_scores


def xgb_full_loop():
    for ordinal, continuous, categorical, data_path in [
        x.values()
        for x in [
            DI1_FEATURES, DI2_FEATURES, DU1_FEATURES, DU2_FEATURES
        ]
    ]:
        file = data_path.split('.')[0].split('/')[-1]
        data = pd.read_csv(data_path)

        # when dealing with data with no missing value
        if 'unimputed' not in data_path:
            data = data.dropna()
            preprocessor = ColumnTransformer(
                transformers=[
                    ('onehot', OneHotEncoder(), categorical),
                    ('std', StandardScaler(), continuous),
                    ('ordinal', OrdinalEncoder(), ordinal)
                ]
            )
        else:
            continuous_transformer = Pipeline(
                steps=[
                    ('continuous_imputer', SimpleImputer(strategy='constant', fill_value=np.nan)),
                    ('scaler', StandardScaler())
                ]
            )
            categorical_transformer = Pipeline(
                steps=[
                    ('categorical_transformer', CategoricalFeatureTransformer(columns=categorical)),
                    ('categorical_imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('encoder', OneHotEncoder())
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ('onehot', categorical_transformer, categorical),
                    ('std', continuous_transformer, continuous),
                    ('ordinal', OrdinalEncoder(), ordinal),
                ]
            )

        x, y = data.drop(columns=['TARGET']), data.TARGET

        xgb_params = {
            'max_depth': [2, 4, 8, 16],
            'n_estimators': [160, 320, 500, 1600, 2400]
        }

        _ = train_xgboost_model_weighted(
            x, y,
            preprocessor,
            xgb_params,
            roc_auc_score,
            'results/xgb_model_{}'.format(file)
        )
