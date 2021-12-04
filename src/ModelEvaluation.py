# Importing packages
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
import shap
from src.DataProcessing import (
    load_obj,
    FeatureSelector,
    CategoricalFeatureTransformer
)
from matplotlib import pyplot as plt
from src.constants import (
    DATA_FILES
)
from sklearn.model_selection import train_test_split


def analyze_confusion_matrix(m):
    total = np.sum(m)
    recall = m[1, 1] / (m[1, 1] + m[1, 0])
    precision = m[1, 1] / (m[1, 1] + m[0, 1])
    accuracy = (m[0, 0] + m[1, 1]) / total
    f_score = 2 * precision * recall / (precision + recall)
    return accuracy, f_score, recall, precision


def process_logistic_hgb_rf_results():
    scores = {}
    stds = {}
    for name in [
        'logistic_regression_model',
        'hist_gradient_boosting_model',
        'random_forest_model'
    ]:
        print('...Processing results from {}...'.format(name))
        di1_res = load_obj('results/{}_data_imputed_1'.format(name))
        di2_res = load_obj('results/{}_data_imputed_2'.format(name))
        di1_res_scores = [di1_res[x]['optimal_score'] for x in di1_res]
        di2_res_scores = [di2_res[x]['optimal_score'] for x in di2_res]
        print('...Processing training results...')
        di1_mean = np.array(di1_res_scores).mean(axis=0)
        di1_std = np.array(di1_res_scores).std(axis=0)
        di2_mean = np.array(di2_res_scores).mean(axis=0)
        di2_std = np.array(di2_res_scores).std(axis=0)
        print('...Formulating scores...')
        scores[
            '{}_data_imputed_1'.format(name)
        ] = di1_mean
        scores[
            '{}_data_imputed_2'.format(name)
        ] = di2_mean
        stds[
            '{}_data_imputed_1'.format(name)
        ] = di1_std
        stds[
            '{}_data_imputed_2'.format(name)
        ] = di2_std
        print('...Done processing {}...'.format(name))
    return scores, stds


def process_xgb_model() -> tuple:
    """
    Calculating train, validation and test scores from saved file, as well as feature importance and SHAP values
    :return: ()
    """
    test_score_means = {}
    test_score_stds = {}
    train_score_means = {}
    train_score_stds = {}
    val_score_means = {}
    val_score_stds = {}
    f_importance = defaultdict(dict)
    s_val = {}
    for data in [
        'data_imputed_1',
        'data_imputed_2',
        'data_unimputed_1',
        'data_unimputed_2'
    ]:
        print('...Processing results from XGB model trained via {}...'.format(data))
        res = load_obj('results/extreme_gradient_boosting_model_{}'.format(data))
        scores = [
            analyze_confusion_matrix(
                res[x]['test_score']
            )
            for x in res
        ]
        # get mean and standard deviations on test scores (vary by random states)
        test_means = np.array(scores).mean(axis=0)
        test_stds = np.array(scores).std(axis=0)
        test_score_means['extreme_gradient_boosting_model_{}'.format(data)] = test_means
        test_score_stds['extreme_gradient_boosting_model_{}'.format(data)] = test_stds

        print('...Packing training, validation and testing scores...')
        # get mean and standard deviations on train and val scores
        train_means = np.array(
            [res[x]['train_scores'] for x in res]
        ).mean(axis=0)
        train_stds = np.array(
            [res[x]['train_scores'] for x in res]
        ).std(axis=0)
        train_score_means['extreme_gradient_boosting_model_{}'.format(data)] = train_means
        train_score_stds['extreme_gradient_boosting_model_{}'.format(data)] = train_stds

        val_means = np.array(
            [res[x]['val_scores'] for x in res]
        ).mean(axis=0)
        val_stds = np.array(
            [res[x]['val_scores'] for x in res]
        )
        val_score_means['extreme_gradient_boosting_model_{}'.format(data)] = val_means
        val_score_stds['extreme_gradient_boosting_model_{}'.format(data)] = val_stds

        print('...Calculating global feature importance...')
        criteria = [x[0] for x in scores]
        optimal_idx = criteria.index(max(criteria))  # highest accuracy as optimal model
        optimal_rs = [x for x in res][optimal_idx]
        optimal_pipeline = res[optimal_rs]['optimal_model_object']

        print('...Reconstructing feature space mapping...')
        if 'unimputed' in data:
            categorical = optimal_pipeline.steps[0][-1].transformers_[0][1].steps[-1][1].get_feature_names_out(
                DATA_FILES[data]['CATEGORICAL_FEATURES']
            )
        else:
            categorical = optimal_pipeline.steps[0][1].transformers_[0][1].get_feature_names_out(
                DATA_FILES[data]['CATEGORICAL_FEATURES']
            )
        feature_names = \
            categorical.tolist() + DATA_FILES[data]['CONTINUOUS_FEATURES'] + DATA_FILES[data]['ORDINAL_FEATURES']
        selected_features = [
            feature_names[i]
            for i in optimal_pipeline.steps[1][1].columns
        ]
        feature_mapping = {
            'f{}'.format(i): x
            for i, x in enumerate(selected_features)
        }

        # calculating feature importance (global)
        print('...Calculating global feature importance...')
        for metric in [
            'weight', 'gain', 'cover', 'total_gain', 'total_cover'
        ]:
            s = optimal_pipeline.steps[-1][1].get_booster().get_score(importance_type=metric)
            f_importance['extreme_gradient_boosting_model_{}'.format(data)][metric] = {
                feature_mapping[x]: s[x]
                for x in s
            }

        # calculating feature importance (local)
        print('...Calculating local feature importance...')
        d = pd.read_csv(DATA_FILES[data]['DATA_PATH'])
        if 'unimputed' not in data:
            d = d.dropna()
        x, y = d.drop(columns=['TARGET']), d.TARGET
        # reproducing the splitting process
        _, x_test, _, _ = train_test_split(
            x, y,
            test_size=0.1,
            stratify=y,
            random_state=optimal_rs
        )
        explainer = shap.TreeExplainer(optimal_pipeline[2])
        x_test = optimal_pipeline[0].transform(x_test)
        x_test = optimal_pipeline[1].transform(x_test)
        shap_val = explainer.shap_values(x_test)
        s_val['extreme_gradient_boosting_model_{}'.format(data)] = (
            selected_features, x_test, shap_val
        )
    return (
        train_score_means,
        train_score_stds,
        val_score_means,
        val_score_stds,
        test_score_means,
        test_score_stds,
        f_importance,
        s_val
    )


def process_feature_importance(scores, score_name, top_n=20, fsize=(12, 10)):
    rank = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    f, s = [x[0] for x in rank], [x[1] for x in rank]
    fig, ax = plt.subplots(figsize=fsize)
    ax.bar(f, s)
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_tick_params(labelrotation=90, pad=5)
    ax.yaxis.set_tick_params(pad=10)
    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)
    # Show top values
    ax.invert_xaxis()
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')
    # plt.ylabel('Top {} feature importance'.format(top_n))
    # plt.xlabel('Scoring metric: {}'.format(score_name))
    # Add Plot Title
    ax.set_title('Optimal Predictive Model Top {} Feature Importance Scoring via {}'.format(top_n, score_name))
    print('...Saving plot to local at ./figures...')
    fig.show()
    fig.savefig('figures/XGB_model_feature_importance_{}'.format(score_name))
    return f



