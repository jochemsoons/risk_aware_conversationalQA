#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a model using parlai's standard training loop.

For documentation, see parlai.scripts.train_model.
"""
import sys
# IMPORTANT: Append your local path to the ParlAi folder
sys.path.append('/YOUR_PATH_TO/ParlAI')

from parlai.scripts.train_model import TrainModel

if __name__ == '__main__':
    TrainModel.main()
