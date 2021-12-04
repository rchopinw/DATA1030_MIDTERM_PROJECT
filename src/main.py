from src.DataProcessing import (
    DetermineSparse,
    FeatureSelector,
    CategoricalFeatureTransformer
)
import xgboost as xgb
import shap
from matplotlib import pyplot as plt
from src.ExtremeBoostingModel import xgb_full_loop
from src.RandomForestModel import random_forest_full_loop
from src.HistGradientBoostingModel import hist_gradient_boosting_full_loop
from src.LogisticRegressionModel import logistic_regression_full_loop
from collections import defaultdict
from src.ModelEvaluation import (
    process_xgb_model,
    process_logistic_hgb_rf_results,
    process_feature_importance
)


if __name__ == '__main__':
    # training and write the models to local: results
    logistic_regression_full_loop()
    hist_gradient_boosting_full_loop()
    random_forest_full_loop()
    xgb_full_loop()

    # evaluating and comparing
    test_scores, test_stds = process_logistic_hgb_rf_results()
    _, _, _, _, xgb_test_scores, xgb_test_stds, f_importance, s_v = process_xgb_model()

    # visualization of feature importance (global and local)
    count = defaultdict(int)
    optimal = 'extreme_gradient_boosting_model_data_unimputed_1'
    for k in f_importance[optimal]:
        f = process_feature_importance(
            scores=f_importance[optimal][k],
            score_name=k,
            fsize=(15, 12)
        )
        for feature in f:
            count[feature] += 1
    most_selected = sorted(
        count,
        key=lambda x: count[x],
        reverse=True
    )[:5]
    print('Mostly selected features are {}'.format(most_selected))
    for f in most_selected:
        shap.dependence_plot(f, s_v[optimal][2], s_v[optimal][1], feature_names=s_v[optimal][0], show=False)
        plt.savefig('figures/{}_shap_visualization.png'.format(f))

