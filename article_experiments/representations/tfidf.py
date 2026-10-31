from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def build_tfidf_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                ),
            ),
        ]
    )


def tune_tfidf(
    x_train,
    y_train,
    seed: int,
):
    """
    Подбор параметров выполняется только внутри outer train.

    Из-за наличия всего двух объектов класса 1 в outer train
    используется 2-fold stratified CV.
    """

    scorer = make_scorer(
        f1_score,
        labels=[2, 3, 4, 5],
        average="macro",
        zero_division=0,
    )

    inner_cv = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=seed,
    )

    pipeline = build_tfidf_pipeline()

    param_grid = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2),
        ],

        "tfidf__min_df": [
            1,
            2,
            5,
        ],

        "tfidf__sublinear_tf": [
            False,
            True,
        ],

        "classifier__C": [
            0.1,
            1.0,
            10.0,
        ],

        "classifier__class_weight": [
            None,
            "balanced",
        ],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )

    search.fit(
        x_train,
        y_train,
    )

    return search