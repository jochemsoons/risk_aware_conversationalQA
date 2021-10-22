#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a model using parlai's standard training loop.

For documentation, see parlai.scripts.train_model.
"""
import sys
# sys.path.append('/home/jochemsoons/AI_MSC_UBUNTU/IR2/Repository/conversationalQA-master/ParlAI')
sys.path.append('/home/lcur0071/Reproduction_repo/conversationalQA-master/ParlAI')
from parlai.scripts.train_model import TrainModel

if __name__ == '__main__':
    TrainModel.main()
