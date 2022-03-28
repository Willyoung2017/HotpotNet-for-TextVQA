# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.textvqa.builder import TextVQABuilder
from mmf.datasets.builders.textvqa.dataset import TextVQADataset
from mmf.datasets.builders.textvqa_hotpot.dataset import TextVQAHotpotDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("textvqa_hotpot")
class TextVQAHotpotBuilder(TextVQABuilder):
    def __init__(
        self, dataset_name="textvqa_hotpot", dataset_class=TextVQAHotpotDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/textvqa_hotpot/defaults.yaml"
