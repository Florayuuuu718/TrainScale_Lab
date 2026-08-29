"""Teaching training engine and gradient reducers for Module 06."""

from .model import TinyTransformer, TinyTransformerConfig
from .reducer import (
    BucketReducer,
    BulkReducer,
    PerParameterReducer,
    ReducerStepStats,
    build_bucket_plan,
)

__all__ = [
    "BulkReducer",
    "BucketReducer",
    "PerParameterReducer",
    "ReducerStepStats",
    "TinyTransformer",
    "TinyTransformerConfig",
    "build_bucket_plan",
]
