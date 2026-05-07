"""model56_undirected_q0: undirected model56 variant with q=0 PE."""

from models.registry import register_model
from torch import Tensor

from models.model56 import MagLapEncoder
from models.model56_undirected import Net as UndirectedNet


@register_model("model56_undirected_q0")
class Net(UndirectedNet):
    """Use the undirected graph blocks and add q=0 PE from dir_pe_ml2."""

    def __init__(self, config):
        super().__init__(config)
        self.pe_k = int(getattr(config, "pe_k", 5))
        pe_hidden = min(self.d_model, 128)
        self.ml_encoder = MagLapEncoder(
            k=self.pe_k,
            hidden_dim=pe_hidden,
            output_dim=self.d_model,
        )
        self.ml_encoder.apply(self._init_module_weights)

    def _add_pe(self, x: Tensor, sample) -> Tensor:
        if "dir_pe_ml2" not in sample:
            raise KeyError("model56_undirected_q0 requires sample['dir_pe_ml2']")
        pe = self.ml_encoder(sample["dir_pe_ml2"])
        return x + pe

    def forward(self, sample, static_feature) -> tuple[Tensor, Tensor]:
        ops, adj, reachability = self.get_data(sample, static_feature)

        degree = self._degree_from_adj(adj)
        x = self._embed_features(ops, degree)
        x = self._add_pe(x, sample)
        x, adj, reachability = self._add_cls_token(x, adj, reachability)

        x, aux_loss = self.encoder(x, adj, reachability)
        x = self.post_norm(x)

        graph_feat = self._readout(x)
        predict = self.predictor(graph_feat)
        if self.predict_target != "latency":
            predict = predict + 0.5
        return predict, aux_loss if self.use_aux_loss else 0


__all__ = ["Net"]
