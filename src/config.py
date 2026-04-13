# src/config.py
class Config:
    DATA_PATH = 'data'
    RANDOM_SEED = 42

    # Placeholder
    NUM_USERS = 0
    NUM_ITEMS = 0
    FEATURE_DIM = 0

    # Training Params
    ROUNDS = 500
    LOCAL_EPOCHS = 8
    LR = 0.0003
    EMBEDDING_DIM = 32
    BATCH_SIZE = 256

    # Federated Optimizer
    # FEDAVG | FEDPROX
    FL_ALGO = 'FEDPROX'
    PROX_MU = 0.01

    # Federated Params
    USERS_PER_ROUND = 30

    # Personalization
    ENABLE_PERSONALIZATION = True

    # Privacy Params
    # PLAIN | LDP | CDP
    PRIVACY_MODE = 'CDP'
    DP_SIGMA = 0.002
    CLIP_NORM = 0.005
    ENABLE_ADAPTIVE_DP = True
    DP_SIGMA_MIN = 0.001
    DP_SIGMA_MAX = 0.10
    DP_PROGRESSIVE_DECAY = 0.40
    DP_SPARSITY_BOOST = 0.15

    ATTACK_ENABLED = True

    # Logging
    SAVE_DIR = 'logs'
    TAIL_WINDOW = 50
