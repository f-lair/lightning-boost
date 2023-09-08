# Copyright The Lightning AI team.
# Copyright 2023 Fabrice von der Lehr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Union

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.model_summary import ModelSummary
from lightning.pytorch.utilities.model_summary.model_summary import (
    ModelSummary as Summary,
)
from lightning.pytorch.utilities.model_summary.model_summary import summarize
from lightning.pytorch.utilities.model_summary.model_summary_deepspeed import (
    DeepSpeedSummary,
)

from lightning_boost.systems import BaseSystem


class LightningBoostModelSummary(ModelSummary):
    def _summary(self, trainer: Trainer, system: BaseSystem) -> Union[DeepSpeedSummary, Summary]:
        # adapted from pytorch-lightning implementation:
        # cf. https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/callbacks/model_summary.py
        # - collected only specific modules

        from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy

        if isinstance(trainer.strategy, DeepSpeedStrategy) and trainer.strategy.zero_stage_3:
            return DeepSpeedSummary(system, max_depth=self._max_depth)
        dummy_system = LightningModule()
        for name, model in system.models.items():
            dummy_system.add_module(name, model)
        return summarize(dummy_system, max_depth=self._max_depth)  # type: ignore
