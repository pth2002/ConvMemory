import torch


class MixerConvMemoryEncoder(torch.nn.Module):
    """Lightweight temporal encoder over a short memory window.

    The input shape is `[batch, window, embedding_dim]`. The query embedding is
    used both for feature construction and query-aware pooling.
    """

    def __init__(
        self,
        dim,
        window_size=5,
        kernel_size=3,
        hidden_dim=256,
        token_mlp_dim=32,
        channel_mlp_dim=512,
        type_vocab_size=0,
        output_mode="residual",
        output_gate_init=0.1,
        score_mode="cosine",
        score_gate_init=0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.output_mode = output_mode
        self.score_mode = score_mode
        self.type_embedding = None
        if type_vocab_size:
            self.type_embedding = torch.nn.Embedding(type_vocab_size, dim)

        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(dim * 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.conv_norm = torch.nn.LayerNorm(hidden_dim)
        self.depthwise_conv = torch.nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.pointwise = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv_gate = torch.nn.Parameter(torch.tensor(0.1))

        self.token_norm = torch.nn.LayerNorm(window_size)
        self.token_mlp = torch.nn.Sequential(
            torch.nn.Linear(window_size, token_mlp_dim),
            torch.nn.GELU(),
            torch.nn.Linear(token_mlp_dim, window_size),
        )
        self.token_gate = torch.nn.Parameter(torch.tensor(0.1))

        self.channel_norm = torch.nn.LayerNorm(hidden_dim)
        self.channel_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, channel_mlp_dim),
            torch.nn.GELU(),
            torch.nn.Linear(channel_mlp_dim, hidden_dim),
        )
        self.channel_gate = torch.nn.Parameter(torch.tensor(0.1))

        self.query_proj = torch.nn.Linear(dim, hidden_dim)
        self.attn_x = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_q = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_v = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.output_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, dim),
            torch.nn.LayerNorm(dim),
        )
        self.output_gate = torch.nn.Parameter(torch.tensor(float(output_gate_init)))
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(dim * 4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.score_gate = torch.nn.Parameter(torch.tensor(float(score_gate_init)))

    def _token_mix(self, h):
        length = h.shape[1]
        if length < self.window_size:
            pad = torch.zeros(
                h.shape[0],
                self.window_size - length,
                h.shape[2],
                dtype=h.dtype,
                device=h.device,
            )
            h_for_mix = torch.cat([h, pad], dim=1)
        else:
            h_for_mix = h[:, : self.window_size]

        mixed = h_for_mix.transpose(1, 2)
        mixed = self.token_mlp(self.token_norm(mixed)).transpose(1, 2)
        return mixed[:, :length]

    def forward(self, x, query=None, type_ids=None):
        base_x = x
        if self.type_embedding is not None and type_ids is not None:
            x = x + self.type_embedding(type_ids)
            base_x = x
        if query is None:
            query = x.mean(dim=1)

        query_norm = torch.nn.functional.normalize(query, dim=-1)
        base_norm = torch.nn.functional.normalize(base_x, dim=-1)
        base_scores = (base_norm * query_norm[:, None, :]).sum(dim=-1)
        base_weights = torch.softmax(base_scores, dim=1)
        base = (base_x * base_weights[:, :, None]).sum(dim=1)

        query_per_turn = query[:, None, :].expand(-1, x.shape[1], -1)
        features = torch.cat([x, x * query_per_turn, torch.abs(x - query_per_turn)], dim=-1)
        h = self.input_proj(features)

        conv_in = self.conv_norm(h).transpose(1, 2)
        conv_out = self.depthwise_conv(conv_in).transpose(1, 2)
        h = h + self.conv_gate * self.pointwise(torch.nn.functional.gelu(conv_out))

        h = h + self.token_gate * self._token_mix(h)
        h = h + self.channel_gate * self.channel_mlp(self.channel_norm(h))

        qh = self.query_proj(query)
        attn = self.attn_v(torch.tanh(self.attn_x(h) + self.attn_q(qh)[:, None, :])).squeeze(-1)
        weights = torch.softmax(attn, dim=1)
        pooled = (h * weights[:, :, None]).sum(dim=1)

        out = self.output_head(
            torch.cat([pooled, qh, pooled * qh, torch.abs(pooled - qh)], dim=-1)
        )
        if self.output_mode == "residual":
            out = base + self.output_gate * out
        return torch.nn.functional.normalize(out, dim=-1)

    def score_windows(self, x, query=None, type_ids=None):
        vectors = self.forward(x, query=query, type_ids=type_ids)
        if query is None:
            query = x.mean(dim=1)
        query_norm = torch.nn.functional.normalize(query, dim=-1)
        cosine = (vectors * query_norm).sum(dim=-1)
        if self.score_mode == "cosine":
            return cosine

        features = torch.cat(
            [vectors, query_norm, vectors * query_norm, torch.abs(vectors - query_norm)],
            dim=-1,
        )
        correction = torch.tanh(self.score_head(features).squeeze(-1))
        return cosine + self.score_gate * correction
