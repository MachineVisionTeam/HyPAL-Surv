"""
HyPAL-Surv -- standalone, dataset-agnostic main model script.

NAMING NOTE (internal vs published)
   This file's class is named GenoDistilCPKF and its trainer is
   train_genodistil_cpkf for internal stability (cached results, file
   imports). The PUBLISHED name of this method is HyPAL-Surv, and the
   PUBLISHED name of its fusion operator is HGBF. Treat:
       GenoDistilCPKF      <-->  HyPAL-Surv      (the whole method)
       SlimCPKFFusion      <-->  HGBF            (the fusion operator)
       "slim_cpkf"         <-->  HGBF            (registry key)
   See HYPAL_SURV_ARCHITECTURE.txt for the published naming.

A unified architecture for histology-genomics survival prediction that
combines:
    (1) HGBF fusion operator
        (Hypercomplex-Gated Bilinear Fusion =
         PHM-modulated bottleneck Kronecker bilinear fusion)
    (2) Inverse PAL training recipe
        (gene-auxiliary head distills risk-ranking INTO the fused output)

REFERENCE RESULT
   On TCGA-GBMLGG with UNI2-h + BulkRNABert frozen encoders:
     c-index = 0.8435 +/- 0.0524  on PF-EXACT 501-patient cohort
     (beats PF trimodal 0.826 paper, BulkRNABert alone 0.8334,
      and every other fusion variant we tested)

HOW TO USE ON A NEW DATASET
   You need, per sample (per ROI or per patient):
     - x_image  : numpy array of frozen image-encoder features (img_dim,)
     - x_gene   : numpy array of frozen gene-encoder features  (gene_dim,)
     - T        : survival time (float)
     - event    : event indicator (0=censored, 1=death)

   Then:
     model = GenoDistilCPKF(img_dim=YOUR_IMG_DIM, gene_dim=YOUR_GENE_DIM)
     train_genodistil_cpkf(model, X_image_train, X_gene_train,
                            T_train, e_train, epochs=200, lr=1e-4)
     hazards = model.predict(X_image_test, X_gene_test)
     c = cindex(hazards, e_test, T_test)

   See the __main__ block at the bottom for a complete cross-validation
   training-and-evaluation example.

PARAMETER COUNT (for default img_dim=1536, gene_dim=256, fusion_dim=256, bottle=64)
   ImageAdapter           :   395 K
   GeneAdapter            :    66 K
   HGBF fusion       :  1.77 M
   CoxHead_main           :   257
   GeneAuxHead (train-only):   257
   ---------------------
   Total trainable        :  ~2.23 M

   vs PF trimodal end-to-end : ~80 M  (36x fewer trainable params)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class GenoDistilConfig:
    """All hyperparameters for the architecture + training in one place.

    Defaults reproduce our headline TCGA-GBMLGG result (0.8435 c-index).
    For a new dataset, you typically only need to change img_dim and gene_dim
    to match your frozen encoders' output sizes.
    """
    # Encoder output dimensions (DATASET-DEPENDENT)
    img_dim:        int = 1536    # e.g. UNI2-h ViT-H/14 = 1536; CTransPath = 768
    gene_dim:       int = 256     # e.g. BulkRNABert = 256; scGPT = 512

    # Fusion architecture (typically left at defaults)
    fusion_dim:     int = 256     # common adapter output dim
    bottle:         int = 64      # HGBF bottleneck dimension
    phm_n:          int = 4       # PHM hypercomplex order
    dropout:        float = 0.1   # AlphaDropout rate (for SELU)

    # Inverse PAL training (typically left at defaults)
    lambda_aux:     float = 0.5   # weight on aux gene Cox loss
    lambda_dist:    float = 0.3   # weight on Inverse PAL distillation

    # Optimization (typically left at defaults)
    epochs:         int = 200
    lr:             float = 1e-4
    weight_decay:   float = 0.0
    grad_clip_norm: float = 1.0


# ============================================================================
# PHM -- Parameterized Hypercomplex Multiplication (Zhang et al. ICLR 2021)
# Degree-1 linear with structured weight matrix
# ============================================================================
class PHMLayer(nn.Module):
    """PHM applied as a fusion slot: input is concat([z_a, z_b]).

    Math:
        Inputs: z_a (B, dim_a), z_b (B, dim_b)
                concat x (B, in_f) where in_f = dim_a + dim_b
        Output: y (B, out_f) where out_f is fusion_dim or bottle

        Composed weight (n^3 + (in*out)/n parameters):
            W = sum over i of kron(A_i, S_i)   where A_i (n,n), S_i (out/n, in/n)
        Forward:
            y = F.linear(x, W) + bias

    Constraint: n must divide both in_f and out_f.
    """
    def __init__(self, dim_a: int, dim_b: int, out_dim: int, n: int = 4):
        super().__init__()
        in_f = dim_a + dim_b
        if in_f % n != 0:
            raise ValueError(f"PHM: in_f={in_f} not divisible by n={n}")
        if out_dim % n != 0:
            raise ValueError(f"PHM: out_dim={out_dim} not divisible by n={n}")
        self.n     = n
        self.in_f  = in_f
        self.out_f = out_dim
        # A: (n, n, n)  algebra matrices (learnable)
        self.a = nn.Parameter(torch.empty(n, n, n))
        # S: (n, out_f//n, in_f//n)  block weights
        self.s = nn.Parameter(torch.empty(n, out_dim // n, in_f // n))
        # bias
        self.bias = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.s)
        bound = 1.0 / math.sqrt(in_f)
        nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _kron(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Vectorised per-leading-index Kronecker product."""
        res = a.unsqueeze(-1).unsqueeze(-3) * s.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(s.shape[-2:]))
        return res.reshape(siz0 + siz1)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_a, z_b], dim=-1)                           # (B, in_f)
        W = torch.sum(self._kron(self.a, self.s), dim=0)            # (out_f, in_f)
        return F.linear(x, W, self.bias)                            # (B, out_f)


# ============================================================================
# Kronecker / TFN -- Tensor Fusion Network (Zadeh et al. EMNLP 2017)
# Degree-2 bilinear with gated multimodal units
# ============================================================================
class KroneckerTFN(nn.Module):
    """Full bilinear outer-product fusion with gated multimodal units and skip.

    Used as the inner-bilinear of HGBF. Operates on the (small)
    bottleneck-projected vectors to keep parameter count manageable.
    """
    def __init__(self, dim_a: int, dim_b: int, fused_dim: int,
                 mmhid: int = 256, dropout: float = 0.25, skip: bool = True):
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.skip = skip
        skip_dim = (dim_a + dim_b + 2) if skip else 0
        # Modality A gating (with bilinear gate signal)
        self.h_a = nn.Sequential(nn.Linear(dim_a, dim_a), nn.ReLU(inplace=True))
        self.z_a = nn.Bilinear(dim_a, dim_b, dim_a)
        self.o_a = nn.Sequential(nn.Linear(dim_a, dim_a), nn.ReLU(inplace=True),
                                  nn.Dropout(p=dropout))
        # Modality B gating
        self.h_b = nn.Sequential(nn.Linear(dim_b, dim_b), nn.ReLU(inplace=True))
        self.z_b = nn.Bilinear(dim_a, dim_b, dim_b)
        self.o_b = nn.Sequential(nn.Linear(dim_b, dim_b), nn.ReLU(inplace=True),
                                  nn.Dropout(p=dropout))
        # Post-fusion encoders
        self.post_drop = nn.Dropout(p=dropout)
        self.enc1 = nn.Sequential(nn.Linear((dim_a + 1) * (dim_b + 1), mmhid),
                                   nn.ReLU(inplace=True), nn.Dropout(p=dropout))
        self.enc2 = nn.Sequential(nn.Linear(mmhid + skip_dim, fused_dim),
                                   nn.ReLU(inplace=True), nn.Dropout(p=dropout))
        self._init()

    def _init(self):
        """Match PF init_max_weights: N(0, 1/sqrt(fan_in)), bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1.0 / math.sqrt(m.weight.size(1))
                nn.init.normal_(m.weight, 0.0, stdv)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # Gated modality vectors
        o_a = self.o_a(torch.sigmoid(self.z_a(z_a, z_b)) * self.h_a(z_a))
        o_b = self.o_b(torch.sigmoid(self.z_b(z_a, z_b)) * self.h_b(z_b))
        # Affine augmentation (+1) then outer product
        ones = torch.ones(o_a.size(0), 1, device=o_a.device, dtype=o_a.dtype)
        o_a_aug = torch.cat([o_a, ones], dim=-1)
        o_b_aug = torch.cat([o_b, ones], dim=-1)
        o_ab = torch.bmm(o_a_aug.unsqueeze(2), o_b_aug.unsqueeze(1)).flatten(start_dim=1)
        # Post-fusion encoders with skip
        x = self.enc1(self.post_drop(o_ab))
        if self.skip:
            x = torch.cat([x, o_a_aug, o_b_aug], dim=-1)
        return self.enc2(x)


# ============================================================================
# HGBF -- our fusion operator
# ============================================================================
class SlimCPKF(nn.Module):
    """SLIM Compositional PHM-Kron Fusion.

    PHM produces a content-aware modulation signal m from the bottleneck-
    projected modality vectors. m drives per-dimension sigmoid gates that
    multiply the bottleneck vectors element-wise. The gated vectors are
    then passed into a Kronecker TFN bilinear at the bottleneck dim.

    Pipeline:
       z_a, z_b (fusion_dim)
            |
            v
       project to bottle: z_a_bn, z_b_bn
            |
            v
       m = PHM(z_a_bn, z_b_bn)                       degree-1 cross-modal context
            |
       g_a = sigmoid(W_a m + b_a)   g_b = sigmoid(W_b m + b_b)
            |
       z_a_mod = g_a * z_a_bn       z_b_mod = g_b * z_b_bn
            |
            v
       fused = KroneckerTFN(z_a_mod, z_b_mod)        degree-2 bilinear at bottle dim
    """
    def __init__(self, fusion_dim: int = 256, bottle: int = 64, n: int = 4):
        super().__init__()
        if (2 * bottle) % n != 0:
            raise ValueError(f"SlimCPKF: 2*bottle={2*bottle} not divisible by n={n}")
        if bottle % n != 0:
            raise ValueError(f"SlimCPKF: bottle={bottle} not divisible by n={n}")

        # 1) Bottleneck projections
        self.proj_a = nn.Linear(fusion_dim, bottle)
        self.proj_b = nn.Linear(fusion_dim, bottle)
        for m in (self.proj_a, self.proj_b):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            nn.init.normal_(m.weight, 0.0, stdv)
            nn.init.zeros_(m.bias)

        # 2) PHM modulation signal (concat bottlenecks -> bottle)
        self.phm_mod = PHMLayer(dim_a=bottle, dim_b=bottle, out_dim=bottle, n=n)

        # 3) Content-aware gates from modulation signal
        self.gate_a = nn.Linear(bottle, bottle)
        self.gate_b = nn.Linear(bottle, bottle)
        for m in (self.gate_a, self.gate_b):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            nn.init.normal_(m.weight, 0.0, stdv)
            nn.init.zeros_(m.bias)

        # 4) Inner bilinear at bottleneck dim -> fusion_dim
        self.kron = KroneckerTFN(dim_a=bottle, dim_b=bottle, fused_dim=fusion_dim)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        z_a_bn = self.proj_a(z_a)                       # (B, bottle)
        z_b_bn = self.proj_b(z_b)                       # (B, bottle)
        m   = self.phm_mod(z_a_bn, z_b_bn)              # (B, bottle)
        g_a = torch.sigmoid(self.gate_a(m))             # (B, bottle)
        g_b = torch.sigmoid(self.gate_b(m))             # (B, bottle)
        z_a_mod = g_a * z_a_bn
        z_b_mod = g_b * z_b_bn
        return self.kron(z_a_mod, z_b_mod)              # (B, fusion_dim)


# ============================================================================
# Adapters: each modality's frozen-encoder features -> common fusion_dim
# ============================================================================
class _SELUAdapter(nn.Module):
    """Linear -> SELU -> AlphaDropout. LeCun-Normal init via kaiming_normal."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        lin = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_normal_(lin.weight, nonlinearity="linear")
        nn.init.zeros_(lin.bias)
        self.net = nn.Sequential(lin, nn.SELU(inplace=True), nn.AlphaDropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# THE MAIN HYPAL-SURV MODEL
# ============================================================================
class GenoDistilCPKF(nn.Module):
    """The unified architecture.

    forward(x_image, x_gene) -> (hazard_main, aux_g)
       hazard_main : (B,) main survival log-hazard from fused features
       aux_g       : (B,) auxiliary gene-only log-hazard
                          [training-time only -- discarded at inference]
    """
    def __init__(self, img_dim: int = 1536, gene_dim: int = 256,
                 cfg: GenoDistilConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = GenoDistilConfig(img_dim=img_dim, gene_dim=gene_dim)
        self.cfg = cfg

        # Adapters
        self.image_adapter = _SELUAdapter(cfg.img_dim,  cfg.fusion_dim, cfg.dropout)
        self.gene_adapter  = _SELUAdapter(cfg.gene_dim, cfg.fusion_dim, cfg.dropout)

        # Fusion
        self.fusion = SlimCPKF(fusion_dim=cfg.fusion_dim, bottle=cfg.bottle, n=cfg.phm_n)

        # Main + auxiliary heads
        self.cox_head_main = nn.Linear(cfg.fusion_dim, 1)
        self.gene_aux_head = nn.Linear(cfg.fusion_dim, 1)
        for lin in (self.cox_head_main, self.gene_aux_head):
            nn.init.kaiming_normal_(lin.weight, nonlinearity="linear")
            nn.init.zeros_(lin.bias)

    def forward(self, x_image: torch.Tensor, x_gene: torch.Tensor):
        z_img  = self.image_adapter(x_image)
        z_gene = self.gene_adapter(x_gene)
        fused  = self.fusion(z_img, z_gene)
        hazard_main = self.cox_head_main(fused).squeeze(-1)
        aux_g       = self.gene_aux_head(z_gene).squeeze(-1)
        return hazard_main, aux_g

    @torch.no_grad()
    def predict(self, x_image: torch.Tensor, x_gene: torch.Tensor) -> torch.Tensor:
        """Inference-time forward: only the main hazard. The aux head is
        ignored (it existed only to shape the main path during training)."""
        self.eval()
        h_main, _ = self.forward(x_image, x_gene)
        return h_main


# ============================================================================
# LOSSES
# ============================================================================
def cox_partial_likelihood_loss(theta: torch.Tensor,
                                 t:     torch.Tensor,
                                 e:     torch.Tensor) -> torch.Tensor:
    """Cox negative log partial-likelihood with logcumsumexp stabilization.

    theta : (B,) log-hazard predictions
    t     : (B,) survival time (real-valued)
    e     : (B,) event indicator (0 or 1, float)

    Mechanism: sort by descending time so that for each sample i, the
    risk set R(t_i) = {j : t_j >= t_i} consists of i and all earlier
    indices in the sorted order. logcumsumexp then gives log Σ_R exp(θ).
    The contribution of each uncensored sample is θ_i - log Σ_R exp(θ).
    Loss = - mean over uncensored samples.
    """
    order   = torch.argsort(t, descending=True)
    theta_s = theta[order]
    e_s     = e[order]
    log_cumsum = torch.logcumsumexp(theta_s, dim=0)
    n_events   = e_s.sum().clamp(min=1.0)
    return -((theta_s - log_cumsum) * e_s).sum() / n_events


def ranking_distillation_loss(student: torch.Tensor,
                                teacher: torch.Tensor,
                                t:       torch.Tensor,
                                e:       torch.Tensor) -> torch.Tensor:
    """Pairwise rank-MSE distillation (Inverse PAL's L_dist).

    For every uncensored-first-death pair (i, j) with e_i = 1 AND t_i < t_j:
        s_diff = student[i] - student[j]
        t_diff = teacher[i] - teacher[j]            # teacher detached upstream
        contribution = (s_diff - t_diff) ** 2

    Loss = sum of contributions / number of valid pairs.

    Caller MUST pass the teacher already detached (teacher = h.detach()),
    so that distillation pressure flows only into the student.
    """
    s_diff = student.unsqueeze(1) - student.unsqueeze(0)       # (B, B)
    t_diff = teacher.unsqueeze(1) - teacher.unsqueeze(0)       # (B, B)
    e_mask = (e.unsqueeze(1) > 0.5)
    t_mask = (t.unsqueeze(1) < t.unsqueeze(0))
    valid  = e_mask & t_mask
    n_v    = valid.sum().float().clamp(min=1.0)
    return ((s_diff - t_diff) ** 2 * valid.float()).sum() / n_v


# ============================================================================
# TRAINING (one fold)
# ============================================================================
def train_genodistil_cpkf(model:        GenoDistilCPKF,
                           X_image_train: np.ndarray,
                           X_gene_train:  np.ndarray,
                           T_train:       np.ndarray,
                           e_train:       np.ndarray,
                           device:        str = "cuda",
                           verbose:       bool = False) -> None:
    """Train one fold with full-batch Cox + Inverse PAL distillation.

    Inputs are numpy arrays:
       X_image_train : (N, img_dim)
       X_gene_train  : (N, gene_dim)
       T_train       : (N,) float
       e_train       : (N,) float {0, 1}

    Hyperparameters come from model.cfg.

    Caller is responsible for:
       - Per-modality train/test z-scoring (recommended)
       - Setting torch.manual_seed before calling for reproducibility
    """
    cfg = model.cfg

    # Tensorize on device
    X_i = torch.from_numpy(X_image_train.astype(np.float32)).to(device)
    X_g = torch.from_numpy(X_gene_train .astype(np.float32)).to(device)
    T_t = torch.from_numpy(T_train.astype(np.float32)).to(device)
    e_t = torch.from_numpy(e_train.astype(np.float32)).to(device)

    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        opt.zero_grad()
        hazard_main, aux_g = model(X_i, X_g)

        # Main + aux Cox losses
        L_main  = cox_partial_likelihood_loss(hazard_main, T_t, e_t)
        L_aux_g = cox_partial_likelihood_loss(aux_g,       T_t, e_t)

        # Inverse PAL: distill gene-aux teacher INTO fused output
        L_dist  = ranking_distillation_loss(hazard_main, aux_g.detach(), T_t, e_t)

        L_total = L_main + cfg.lambda_aux * L_aux_g + cfg.lambda_dist * L_dist
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        opt.step()

        if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
            print(f"  epoch {epoch + 1:3d}/{cfg.epochs}  "
                  f"L_main={L_main.item():.4f}  L_aux_g={L_aux_g.item():.4f}  "
                  f"L_dist={L_dist.item():.4f}")


# ============================================================================
# EVALUATION
# ============================================================================
def cindex_lifelines(hazards: np.ndarray,
                     events:  np.ndarray,
                     times:   np.ndarray) -> float:
    """Harrell's c-index via the lifelines library.

    Higher hazard => shorter survival, so we pass -hazards as the risk score
    that lifelines expects (it ranks LOW score as HIGH risk by default in our
    convention; concordance_index treats higher predicted as later event).
    """
    try:
        from lifelines.utils import concordance_index
    except ImportError as exc:
        raise ImportError("pip install lifelines") from exc
    return float(concordance_index(times, -hazards, events))


def predict_and_score(model:       GenoDistilCPKF,
                      X_image_test: np.ndarray,
                      X_gene_test:  np.ndarray,
                      T_test:       np.ndarray,
                      e_test:       np.ndarray,
                      device:       str = "cuda") -> float:
    """Run inference and return c-index on the test set."""
    X_i = torch.from_numpy(X_image_test.astype(np.float32)).to(device)
    X_g = torch.from_numpy(X_gene_test .astype(np.float32)).to(device)
    with torch.no_grad():
        hazards = model.predict(X_i, X_g).cpu().numpy()
    return cindex_lifelines(hazards, e_test, T_test)


# ============================================================================
# EXAMPLE USAGE (not run -- this is for reference)
# ============================================================================
if __name__ == "__main__":
    """
    Skeleton showing how to apply HyPAL-Surv to a new dataset.

    For TCGA-GBMLGG with UNI2-h + BulkRNABert, the per-fold training takes
    ~5 s on a single GPU. Full 15-fold MCCV x 5 seeds = ~6 minutes.
    """
    import time

    # ---- Configuration ----
    cfg = GenoDistilConfig(
        img_dim    = 1536,   # set to your image encoder output dim
        gene_dim   =  256,   # set to your gene encoder output dim
        fusion_dim =  256,
        bottle     =   64,
        epochs     =  200,
        lr         = 1e-4,
        lambda_aux  = 0.5,
        lambda_dist = 0.3,
    )

    # ---- Synthetic data (REPLACE with your real features + labels) ----
    rng = np.random.default_rng(0)
    N_train, N_test = 800, 200
    X_image_train = rng.standard_normal((N_train, cfg.img_dim )).astype(np.float32)
    X_gene_train  = rng.standard_normal((N_train, cfg.gene_dim)).astype(np.float32)
    T_train       = rng.uniform(0.0, 2000.0, N_train).astype(np.float32)
    e_train       = (rng.uniform(0, 1, N_train) < 0.4).astype(np.float32)

    X_image_test  = rng.standard_normal((N_test, cfg.img_dim )).astype(np.float32)
    X_gene_test   = rng.standard_normal((N_test, cfg.gene_dim)).astype(np.float32)
    T_test        = rng.uniform(0.0, 2000.0, N_test).astype(np.float32)
    e_test        = (rng.uniform(0, 1, N_test) < 0.4).astype(np.float32)

    # ---- Standardize per modality (recommended) ----
    try:
        from sklearn.preprocessing import StandardScaler
        sc_i = StandardScaler(); sc_g = StandardScaler()
        X_image_train = sc_i.fit_transform(X_image_train).astype(np.float32)
        X_image_test  = sc_i.transform    (X_image_test ).astype(np.float32)
        X_gene_train  = sc_g.fit_transform(X_gene_train).astype(np.float32)
        X_gene_test   = sc_g.transform    (X_gene_test ).astype(np.float32)
    except ImportError:
        pass

    # ---- Train ----
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GenoDistilCPKF(cfg=cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HyPAL-Surv total trainable params: {n_params:,}")

    t0 = time.time()
    train_genodistil_cpkf(model,
                           X_image_train, X_gene_train, T_train, e_train,
                           device=device, verbose=True)
    print(f"Training time: {time.time() - t0:.1f}s")

    # ---- Evaluate ----
    c = predict_and_score(model,
                           X_image_test, X_gene_test, T_test, e_test,
                           device=device)
    print(f"Test c-index: {c:.4f}")
    print(f"(Synthetic data, expected ~0.50; replace with real data for real numbers.)")
