"""HyPAL-Surv: Hypercomplex-Gated Bilinear Fusion with Inverse
Peer-Assisted Learning for histology-genomics survival prediction.

Public API:
    from hypal_surv.model       import HyPALSurv, train_hypal_surv, GenoDistilConfig
    from hypal_surv.fusion_ops  import HGBF, FUSION_REGISTRY

Note: internal Python class names are GenoDistilCPKF (model) and
SlimCPKFFusion (HGBF operator); these are kept for cached-results
compatibility. They alias to the published names below.
"""
from .model import (
    GenoDistilCPKF as HyPALSurv,
    train_genodistil_cpkf as train_hypal_surv,
    GenoDistilConfig,
)
from .fusion_ops import (
    SlimCPKFFusion as HGBF,
    FUSION_REGISTRY,
)

__all__ = [
    "HyPALSurv",
    "train_hypal_surv",
    "GenoDistilConfig",
    "HGBF",
    "FUSION_REGISTRY",
]
__version__ = "1.0.0"
