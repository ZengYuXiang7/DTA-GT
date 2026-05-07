"""model56_q0: directed model56 with q=0 Magnetic Laplacian PE."""

from models.registry import register_model
from torch import Tensor

from models.model56 import Net as Model56Net


@register_model("model56_q0")
class Net(Model56Net):
    """Keep directed model56 unchanged except reading q=0 PE from dir_pe_ml2."""

    def _add_pe(self, x: Tensor, sample) -> Tensor:
        if "dir_pe_ml2" not in sample:
            raise KeyError("model56_q0 requires sample['dir_pe_ml2']")
        pe = self.ml_encoder(sample["dir_pe_ml2"])
        return x + pe


__all__ = ["Net"]
