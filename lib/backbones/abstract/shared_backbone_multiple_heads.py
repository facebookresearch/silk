# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

import torch
from silk.backbones.silk.coords import (
    CoordinateMappingComposer,
    CoordinateMappingProvider,
)
from silk.flow import AutoForward, Flow
from silk.transforms.abstract import MixedModuleDict


class SharedBackboneMultipleHeads(
    AutoForward,
    torch.nn.Module,
):
    def __init__(
        self,
        backbone,
        input_name: str,
        backbone_output_name: Union[str, Tuple[str]],
    ) -> None:
        torch.nn.Module.__init__(self)
        AutoForward.__init__(
            self,
            Flow(input_name),
            default_outputs=backbone_output_name,
        )

        self._input_name = input_name
        self._backbone_output_name = backbone_output_name
        self._backbone = backbone

        # handle coordinate mappings
        self._coordinate_mappings_composer = CoordinateMappingComposer()
        assert isinstance(self._backbone, CoordinateMappingProvider)

        if self.is_multi_backbone_outputs:
            assert len(self._backbone_output_name) == len(self._backbone.mappings())

            for i, mapping in enumerate(self._backbone.mappings()):
                self._coordinate_mappings_composer.set(
                    self._input_name,
                    self._backbone_output_name[i],
                    mapping,
                )
        else:
            self._coordinate_mappings_composer.set(
                self._input_name,
                self._backbone_output_name,
                self._backbone.mappings(),
            )

        # self._coordinate_mappings: Dict[str, CoordinateMapping],
        self.flow.define_transition(backbone_output_name, self.backbone, input_name)

        self._heads = MixedModuleDict()

    @property
    def coordinate_mapping_composer(self):
        return self._coordinate_mappings_composer

    @property
    def backbone(self):
        return self._backbone

    @property
    def is_multi_backbone_outputs(self):
        return not isinstance(self.backbone_output_name, str)

    @property
    def heads(self):
        return self._heads

    @property
    def head_names(self):
        return tuple(self._heads.keys())

    @property
    def backbone_output_name(self):
        return self._backbone_output_name

    @property
    def input_name(self):
        return self._input_name

    def add_head_to_backbone_output(self, head_name, head, backbone_output_name=None):
        if backbone_output_name is None:
            if self.is_multi_backbone_outputs:
                raise RuntimeError(
                    f"the backbone has {len(self.backbone_output_name)} outputs {self.backbone_output_name} and one should be set using the `backbone_output_name` parameter"
                )
            backbone_output_name = self.backbone_output_name
        elif not isinstance(backbone_output_name, str):
            raise RuntimeError(
                "invalid type for `backbone_output_name` parameter, should be a string"
            )

        # check existing head
        if head_name in self.heads:
            raise RuntimeError(f"head '{head_name}' has already been added")

        if not isinstance(head, CoordinateMappingProvider):
            raise RuntimeError(
                f"head '{head_name}' should sub-class `CoordinateMappingProvider`"
            )

        self.heads[head_name] = head
        self.flow.define_transition(head_name, head, backbone_output_name)
        self._coordinate_mappings_composer.set(
            backbone_output_name,
            head_name,
            head.mappings(),
        )

    def add_head(self, head_name, head):
        self.add_head_to_backbone_output(head_name, head)

    def add_heads(self, **heads):
        # add heads and make them accessible to flow
        for name, head in heads.items():
            self.add_head(name, head)
