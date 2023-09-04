from typing import Optional, Tuple
import neurotorch as nt
import torch


class WilsonCowanCURBDJtoILayer(WilsonCowanCURBDLayer):
    def __init__(*args, **kwargs):
        kwargs["connectivity_convention"] = "j->i"
        super().__init__(*args, **kwargs)

    def build(self) -> 'WilsonCowanCURBDJtoILayer':
        super().build()
        self._forward_weights = torch.nn.Parameter(
            torch.empty((int(self.output_size), int(self.input_size)), device=self.device, dtype=torch.float32),
            requires_grad=self.requires_grad
        )
        if self.force_dale_law:
            self._forward_sign = torch.nn.Parameter(
                torch.empty((1, int(self.input_size)), dtype=torch.float32, device=self.device),
                requires_grad=self.force_dale_law
            )
            if self.use_recurrent_connection:
                self._recurrent_sign = torch.nn.Parameter(
                    torch.empty((1, int(self.output_size)), dtype=torch.float32, device=self.device),
                    requires_grad=self.force_dale_law
                )
        self.initialize_weights_()
        return self

    def create_empty_state(self, batch_size: int = 1, **kwargs) -> Tuple[torch.Tensor, ...]:
        if self.kwargs["hh_init"].lower() == "given":
            assert "h0" in self.kwargs, "h0 must be provided as a tuple of tensors when hh_init is 'given'."
            h0 = self.kwargs["h0"]
            assert isinstance(h0, (tuple, list)), "h0 must be a tuple of tensors."
            state = [to_tensor(h0_, dtype=torch.float32).to(self.device) for h0_ in h0]
        else:
            state = super().create_empty_state(batch_size, **kwargs)
        return tuple(state)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, ...]] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        batch_size, nb_features = inputs.shape

        out_shape = tuple(inputs.shape[:-1]) + (self.forward_weights.shape[0],)  # [*, f_out]
        inputs_view = inputs.view(-1, inputs.shape[-1])  # [*, f_in] -> [B, f_in]
        inputs_permuted = inputs_view.permute(1, 0)  # [B, f_in] -> [f_in, B]

        hh, = self._init_forward_state(state, batch_size, inputs=inputs_view, **kwargs)  # [B, f_out]
        post_activation = self.activation(hh).permute(1, 0)  # [B, f_out] -> [f_out, B]

        if self.use_recurrent_connection:
            # [f_out, f_out] @ [f_out, B] -> [f_out, B]
            rec_inputs = torch.matmul(torch.mul(self.recurrent_weights, self.rec_mask), post_activation)
        else:
            rec_inputs = 0.0

        # [f_out, f_in] @ [f_in, B] -> [f_out, B]
        weighted_current = torch.matmul(self.forward_weights, inputs_permuted)
        jr = (rec_inputs + weighted_current).permute(1, 0)  # [f_out, B] -> [B, f_out]
        next_hh = hh + self.dt * (-hh + jr) / self.tau  # [B, f_out]
        output = post_activation.permute(1, 0).view(out_shape)  # [f_out, B] -> [B, f_out] -> [*, f_out]
        setattr(self, "hh", next_hh)
        return output, (next_hh, )



