from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_linear_learner, default_tree_learner

from src.utils import tr


@dataclass(frozen=True)
class ModelConfig:
    run_name: str
    estimator: Any
    Base: Any
    Dist: Any
    n_estimators: int
    learning_rate: float | int
    random_state: int
    verbose: bool
    pooling: str
    min_train_years: int
    y: str
    X: list[str]

model_configs = {
    'ridge': ModelConfig(
        run_name="ridge",
        estimator=NGBRegressor,
        Base=default_linear_learner,
        Dist=Normal,
        n_estimators=tr['ridge']['n_estimators'],
        learning_rate=tr['ridge']['learning_rate'],
        random_state=tr['ridge']['random_state'],
        verbose=tr['ridge']['verbose'],
        pooling=tr['ridge']['pooling'],
        min_train_years=tr['ridge']['min_train_years'],
        y=tr['ridge']['y'],
        X=tr['ridge']['X'],
    ),

    'base': ModelConfig(
        run_name="base",
        estimator=NGBRegressor,
        Base=default_tree_learner,
        Dist=Normal,
        n_estimators=tr['base']['n_estimators'],
        learning_rate=tr['base']['learning_rate'],
        random_state=tr['base']['random_state'],
        verbose=tr['base']['verbose'],
        pooling=tr['base']['pooling'],
        min_train_years=tr['base']['min_train_years'],
        y=tr['base']['y'],
        X=tr['base']['X'],
    ),

    'main': ModelConfig(
        run_name="main",
        estimator=NGBRegressor,
        Base=default_tree_learner,
        Dist=Normal,
        n_estimators=tr['main']['n_estimators'],
        learning_rate=tr['main']['learning_rate'],
        random_state=tr['main']['random_state'],
        verbose=tr['main']['verbose'],
        pooling=tr['main']['pooling'],
        min_train_years=tr['main']['min_train_years'],
        y=tr['main']['y'],
        X=tr['main']['X'],
    ),
}
