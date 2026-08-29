from __future__ import annotations

import os
import sys
from datetime import timedelta

import torch.distributed as dist
from torch import nn
from trainscale_engine.reducer import BucketReducer

dist.init_process_group("gloo", timeout=timedelta(seconds=20))
try:
    model = nn.Sequential(nn.Linear(4, 4))
    if int(os.environ["RANK"]) == 1:
        model.add_module("rank_one_only", nn.Linear(4, 4))
    try:
        BucketReducer(model, 1024, asynchronous=True)
    except RuntimeError as error:
        if "bucket plan differs across ranks" not in str(error):
            raise
        print(str(error))
    else:
        raise AssertionError("mismatched bucket plans were not rejected")
finally:
    dist.destroy_process_group()

sys.exit(0)
