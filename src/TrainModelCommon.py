from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold


class TrainModel:
    def __init__(
            self, x, y, test_size=0.2, random_state=42, stratify=True,
    ):
        self.x = x
        self.y = y
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state
        self.optimal_res = {}

    def __split_data(
            self, random_state
    ) -> tuple:
        return train_test_split(
            self.x, self.y,
            test_size=self.test_size,
            random_state=random_state,
            shuffle=True,
            stratify=self.y if self.stratify else None
        )

    def train(
            self, model, params,
            k_fold=4,
            num_rs=10,
            random_states=None,
            set_weight=True,
            scoring=None,
            optimal_criteria='roc_auc',
            verbose=True,
    ):
        if random_states is None:
            random_states = [
                x * num_rs for x in range(num_rs)
            ]

        if scoring is None:
            scoring = [
                accuracy_score, f1_score, recall_score, precision_score
            ]

        for r in random_states:
            if verbose:
                print(
                    "Currently executing Grid Search CV with Random State {}.".format(
                        r
                    )
                )
            x_train, x_test, y_train, y_test = self.__split_data(random_state=r)
            kf = StratifiedKFold(
                n_splits=k_fold,
                shuffle=True,
                random_state=r,
            )
            model.set_params(
                model__random_state=r
            )
            if set_weight:
                model.set_params(
                    model__class_weight={
                        1: 1 - sum(y_train)/len(y_train),
                        0: sum(y_train)/len(y_train)
                    }
                )
            gs_cv = GridSearchCV(
                model,
                params,
                cv=kf,
                scoring=optimal_criteria,
                refit=optimal_criteria
            )

            gs_cv.fit(x_train, y_train)

            optimal_model = gs_cv.best_params_

            y_test_pred = gs_cv.best_estimator_.predict(
                x_test
            )
            optimal_score = [f(y_test, y_test_pred) for f in scoring]

            self.optimal_res[r] = {
                'optimal_model': optimal_model,
                'optimal_score': optimal_score,
                'optimal_model_object': gs_cv.best_estimator_
            }

        return self.optimal_res