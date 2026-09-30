from torch import nn

LATENT_DIMS = 128
FEATURE_DIMS = 40

_EMBEDDER_HIDDEN_DIMS = 256

Embedder = nn.Sequential(
    nn.Linear(FEATURE_DIMS, _EMBEDDER_HIDDEN_DIMS),
    nn.ReLU(),
    nn.Linear(_EMBEDDER_HIDDEN_DIMS, _EMBEDDER_HIDDEN_DIMS),
    nn.ReLU(),
    nn.Linear(_EMBEDDER_HIDDEN_DIMS, _EMBEDDER_HIDDEN_DIMS),
    nn.ReLU(),
    nn.Linear(_EMBEDDER_HIDDEN_DIMS, LATENT_DIMS),
)

_DECODER_HIDDEN_DIMS = 256

Decoder = nn.Sequential(
    nn.Linear(LATENT_DIMS, _DECODER_HIDDEN_DIMS),
    nn.ReLU(),
    nn.Linear(_DECODER_HIDDEN_DIMS, _DECODER_HIDDEN_DIMS),
    nn.ReLU(),
    nn.Linear(_DECODER_HIDDEN_DIMS, _DECODER_HIDDEN_DIMS),
    nn.ReLU(),
    nn.Linear(_DECODER_HIDDEN_DIMS, FEATURE_DIMS),
)
