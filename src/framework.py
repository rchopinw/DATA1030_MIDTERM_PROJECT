# Importing packages
import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold


class DetermineSparse:
    def __init__(
            self
    ):
        self.sparse_components = None

    def determine_sparse(
            self, x
    ) -> list:
        if self.sparse_components:
            return self.sparse_components
        self.sparse_components = [
            set(
                np.unique(x[:, i]).astype(int)
            ) == {0, 1}
            for i in range(x.shape[1])
        ]
        return self.sparse_components


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
            scoring=None,
            optimal_criteria='accuracy',
            verbose=True,
    ):
        if random_states is None:
            random_states = [
                x * num_rs for x in range(num_rs)
            ]

        if scoring is None:
            scoring = [
                'accuracy', 'f1', 'recall', 'precision', 'roc_auc'
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
            gs_cv = GridSearchCV(
                model,
                params,
                cv=kf,
                scoring=scoring,
                refit=optimal_criteria
            )

            gs_cv.fit(x_train, y_train)

            optimal_model = gs_cv.best_params_
            optimal_score = gs_cv.score(x_test, y_test)

            self.optimal_res[r] = {
                'optimal_model': optimal_model,
                'optimal_score': optimal_score
            }

        return self.optimal_res
