"""EgmaMath dataset. Context: none, options: text."""

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.registry import register_task


@register_task("egma-math")
class EgmaMathDataset(VLMDataset):
    """Reads raw egma-math corpus and serves text-only forced-choice trials."""

    def _load_trials(self) -> None:
        """Load egma-math trials from corpus CSV and asset index."""
        pass
