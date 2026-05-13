from .encoder import MixerConvMemoryEncoder
from .scoring import CELiteScorer


def build_default_components(
    embedding_dim,
    window_size=5,
    kernel_size=3,
    hidden_dim=256,
    token_mlp_dim=32,
    channel_mlp_dim=512,
    extra_scalar_features=5,
    device="cpu",
):
    conv_model = MixerConvMemoryEncoder(
        embedding_dim,
        window_size=window_size,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        token_mlp_dim=token_mlp_dim,
        channel_mlp_dim=channel_mlp_dim,
        output_mode="residual",
        output_gate_init=0.1,
        score_mode="cosine",
    ).to(device)
    scorer = CELiteScorer(
        embedding_dim,
        hidden_dim=hidden_dim,
        extra_scalar_features=extra_scalar_features,
    ).to(device)
    return conv_model, scorer
