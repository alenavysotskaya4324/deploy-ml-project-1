import hydra
from sklearn.pipeline import Pipeline, FeatureUnion
from src.models.column_selector import ColumnSelector
from omegaconf import DictConfig


def create_model(cfg: DictConfig):
    cat_cols = cfg.data.categorical_columns
    num_col = cfg.data.numerical_columns

    enc = hydra.utils.instantiate(cfg.enc, cols=cat_cols)

    cat_pipe = Pipeline([
        ('selector', ColumnSelector(cat_cols)),
        ('encoder', enc)
    ])

    num_pipe = Pipeline([
        ('selector', ColumnSelector(num_col))
    ])

    preprocessor = FeatureUnion([
        ('cat', cat_pipe),
        ('num', num_pipe)
    ])

    model = hydra.utils.instantiate(cfg.model)

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])

    return pipeline
