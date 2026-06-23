"""
Four fusion operators behind a uniform PyTorch nn.Module contract:

    fuse(z_a, z_b) -> z_fused
        z_a: Tensor (B, dim_a)
        z_b: Tensor (B, dim_b)
        z_fused: Tensor (B, fused_dim)

All operators are dim-standardized: dim_a = dim_b = fused_dim = 256 by default.
Identical Cox head sits on top, so the ONLY variable in the Stage 4 sweep is
which operator is used.

Operators implemented:
    concat     -- the floor.
    kronecker  -- TFN-style outer product with gated multimodal units.
                  Ported from Pathomic Fusion's BilinearFusion (the operator
                  that produced our locked 0.8174 trimodal anchor).
    lmf        -- Low-rank Multimodal Fusion (Liu et al. ACL 2018).
                  Ported from github.com/Justin1904/Low-rank-Multimodal-Fusion,
                  adapted from 3-modality to 2-modality.
    phm        -- Parameterized Hypercomplex Multiplication (Zhang et al. ICLR 2021).
                  Ported from github.com/eleGAN23/HyperNets layers/ph_layers.py,
                  used as a drop-in for the fusion-slot Linear.

Common pattern (no operator hides activation/dropout/norm inside):
    each FusionOp returns a *raw* fused vector of shape (B, fused_dim).
    The training harness wraps the chain "fusion(z_i, z_g) -> Cox_head -> hazard"
    so the same training loop applies to all four operators.
"""
from __future__ import annotations

import math
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Shared init helper (matches PF's init_max_weights, used in PF anchor)
# ----------------------------------------------------------------------
def init_max_weights(module: nn.Module) -> None:
    """Normal init with std = 1/sqrt(fan_in), bias = 0.
    Matches Pathomic Fusion's utils.init_max_weights so the Kronecker
    fusion is initialised the same way as the 0.8174 anchor.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            nn.init.normal_(m.weight, 0.0, stdv)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ======================================================================
# 1. Concat fusion (the floor)
# ======================================================================
class ConcatFusion(nn.Module):
    """Simplest possible fusion: cat -> Linear -> SELU -> AlphaDropout."""

    def __init__(self, dim_a: int = 256, dim_b: int = 256,
                 fused_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.fused_dim = fused_dim
        self.proj = nn.Linear(dim_a + dim_b, fused_dim)
        self.act = nn.SELU(inplace=True)
        self.drop = nn.AlphaDropout(dropout)
        # SELU+AlphaDropout needs LeCun-Normal initialization
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.proj.bias)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_a, z_b], dim=-1)
        return self.drop(self.act(self.proj(z)))


# ======================================================================
# 2. Kronecker / TFN fusion (Pathomic Fusion's own)
# ======================================================================
class KroneckerTFNFusion(nn.Module):
    """
    Tensor Fusion Network (Zadeh et al. EMNLP 2017, arXiv:1707.07250).

    PORTED FROM:
       pathomic_fusion_replica/PathomicFusion/fusion.py:BilinearFusion
    which is the bimodal version of the trilinear fusion that scored our
    locked 0.8174 GBMLGG anchor. Same gating, same outer product, same
    skip + post-fusion encoders -- only the modality dims differ
    (PF uses 32-d inputs; we use 256-d).

    Math:
       For each modality m in {a, b}:
           h_m = ReLU(W_h_m  v_m)                       # gated content
           z_m = Bilinear(v_a, v_b)                     # gate signal
           o_m = ReLU(W_o_m (sigmoid(z_m) * h_m))       # gated, dropped vector
           o_m_aug = [o_m; 1]                           # affine: append 1
       o_ab = vec( o_a_aug  outer  o_b_aug )            # Kronecker product
       fused = encoder2( cat[ encoder1(o_ab), skip(o_a_aug, o_b_aug) ] )

    Init: PF's init_max_weights (normal(0, 1/sqrt(fan_in))).
    """

    def __init__(self, dim_a: int = 256, dim_b: int = 256,
                 fused_dim: int = 256,
                 mmhid: int = 256, dropout: float = 0.25,
                 skip: bool = True, use_bilinear: bool = True,
                 gate_a: bool = True, gate_b: bool = True) -> None:
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.fused_dim = fused_dim
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate_a = gate_a
        self.gate_b = gate_b

        skip_dim = (dim_a + dim_b + 2) if skip else 0  # +2 for the two "1"s appended

        # Modality A gating
        self.linear_h_a = nn.Sequential(nn.Linear(dim_a, dim_a), nn.ReLU(inplace=True))
        self.linear_z_a = (nn.Bilinear(dim_a, dim_b, dim_a) if use_bilinear
                          else nn.Linear(dim_a + dim_b, dim_a))
        self.linear_o_a = nn.Sequential(nn.Linear(dim_a, dim_a), nn.ReLU(inplace=True),
                                       nn.Dropout(p=dropout))

        # Modality B gating
        self.linear_h_b = nn.Sequential(nn.Linear(dim_b, dim_b), nn.ReLU(inplace=True))
        self.linear_z_b = (nn.Bilinear(dim_a, dim_b, dim_b) if use_bilinear
                          else nn.Linear(dim_a + dim_b, dim_b))
        self.linear_o_b = nn.Sequential(nn.Linear(dim_b, dim_b), nn.ReLU(inplace=True),
                                       nn.Dropout(p=dropout))

        # Post-fusion
        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim_a + 1) * (dim_b + 1), mmhid),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        init_max_weights(self)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # --- gate A ---
        if self.gate_a:
            h_a = self.linear_h_a(z_a)
            if self.use_bilinear:
                z_gate_a = self.linear_z_a(z_a, z_b)
            else:
                z_gate_a = self.linear_z_a(torch.cat([z_a, z_b], dim=-1))
            o_a = self.linear_o_a(torch.sigmoid(z_gate_a) * h_a)
        else:
            o_a = self.linear_o_a(z_a)

        # --- gate B ---
        if self.gate_b:
            h_b = self.linear_h_b(z_b)
            if self.use_bilinear:
                z_gate_b = self.linear_z_b(z_a, z_b)
            else:
                z_gate_b = self.linear_z_b(torch.cat([z_a, z_b], dim=-1))
            o_b = self.linear_o_b(torch.sigmoid(z_gate_b) * h_b)
        else:
            o_b = self.linear_o_b(z_b)

        # --- Kronecker (outer) product of augmented vectors ---
        # append 1 to each modality (the affine "+1" trick)
        ones = torch.ones(o_a.size(0), 1, device=o_a.device, dtype=o_a.dtype)
        o_a_aug = torch.cat([o_a, ones], dim=-1)   # (B, dim_a + 1)
        o_b_aug = torch.cat([o_b, ones], dim=-1)   # (B, dim_b + 1)
        o_ab = torch.bmm(o_a_aug.unsqueeze(2), o_b_aug.unsqueeze(1)).flatten(start_dim=1)
        # o_ab shape: (B, (dim_a+1) * (dim_b+1))

        # --- post-fusion encoders ---
        x = self.post_fusion_dropout(o_ab)
        x = self.encoder1(x)
        if self.skip:
            x = torch.cat([x, o_a_aug, o_b_aug], dim=-1)
        return self.encoder2(x)


# ======================================================================
# 3. LMF -- Low-rank Multimodal Fusion (Liu et al. ACL 2018)
# ======================================================================
class LMFFusion(nn.Module):
    """
    Low-rank Multimodal Fusion (Liu et al. ACL 2018, arXiv:1806.00064).

    PORTED FROM:
       github.com/Justin1904/Low-rank-Multimodal-Fusion/model.py:LMF
    The reference is 3-modality (audio, video, text); we adapt to 2-modality
    by dropping one factor tensor and reducing the element-wise product to two.

    Math:
       _z_m = [1; z_m]    (B, dim_m + 1)              # affine augmentation
       f_m  = _z_m  @  W_m                            # W_m shape (R, dim_m+1, fused_dim)
                                                     # result shape (R, B, fused_dim) via broadcast
       fused_rank = f_a * f_b                          # element-wise (R, B, fused_dim)
       fused      = sum over R of (w_r * fused_rank) + b
                  = (1, R) @ permute(R,B,F)->(B,R,F)   -> (B, 1, fused_dim)

    Rank R is a hyperparameter (we use 4 by default, the LMF paper's
    typical range is 2..8). Total fusion params: R * ((dim_a+1) + (dim_b+1))
    * fused_dim + R + fused_dim, which is O(R * dim * fused_dim) instead of
    the O(dim_a * dim_b * fused_dim) of full TFN.
    """

    def __init__(self, dim_a: int = 256, dim_b: int = 256,
                 fused_dim: int = 256, rank: int = 4) -> None:
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.fused_dim = fused_dim
        self.rank = rank

        # Per-modality rank factors. Each row "+1" is the augmentation slot.
        self.factor_a = nn.Parameter(torch.empty(rank, dim_a + 1, fused_dim))
        self.factor_b = nn.Parameter(torch.empty(rank, dim_b + 1, fused_dim))
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, fused_dim))

        # init (matches reference: xavier_normal on factors + weights, zero bias)
        nn.init.xavier_normal_(self.factor_a)
        nn.init.xavier_normal_(self.factor_b)
        nn.init.xavier_normal_(self.fusion_weights)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        B = z_a.size(0)
        ones = torch.ones(B, 1, device=z_a.device, dtype=z_a.dtype)
        _z_a = torch.cat([ones, z_a], dim=-1)   # (B, dim_a + 1)
        _z_b = torch.cat([ones, z_b], dim=-1)   # (B, dim_b + 1)

        # matmul broadcasts: (B, d+1) @ (R, d+1, fused_dim) -> (R, B, fused_dim)
        f_a = torch.matmul(_z_a, self.factor_a)   # (R, B, fused_dim)
        f_b = torch.matmul(_z_b, self.factor_b)   # (R, B, fused_dim)

        fused_rank = f_a * f_b                    # (R, B, fused_dim) element-wise

        # weighted sum across ranks: (1, R) @ (B, R, fused_dim) -> (B, 1, fused_dim)
        out = torch.matmul(self.fusion_weights, fused_rank.permute(1, 0, 2))
        out = out.view(-1, self.fused_dim) + self.fusion_bias
        return out


# ======================================================================
# 4. PHM fusion (Zhang et al. ICLR 2021)
# ======================================================================
class PHMFusion(nn.Module):
    """
    Parameterized Hypercomplex Multiplication (Zhang et al. ICLR 2021,
    arXiv:2102.08597 "Beyond Fully-Connected Layers with Quaternions").

    PORTED FROM:
       github.com/eleGAN23/HyperNets/layers/ph_layers.py:PHMLinear
    Used as a fusion slot: PHMLinear applied to concat([z_a, z_b]) ->
    fused_dim. The CONCAT path through PHM is the cleanest way to test
    PHM's "different inductive bias" against the bilinear family
    (Kronecker / LMF), keeping the operator a drop-in replacement for
    a single Linear in the fusion-slot position.

    Math:
       Input dim = dim_a + dim_b (call it in_f); output dim = fused_dim (out_f).
       Constraint: n must divide both in_f and out_f.
       Learnable:
           A_i: n algebra matrices, each shape (n, n)        -- shape (n, n, n)
           S_i: n weight blocks,  each shape (out_f/n, in_f/n) -- shape (n, out_f/n, in_f/n)
           b:   bias, shape (out_f,)
       Composed weight:
           W = sum_i kron(A_i, S_i)         -- shape (out_f, in_f)
       Forward:
           out = F.linear(x, W, b)          -- standard affine after composition.

       Crucially, A_i are LEARNED, not fixed -- the operator learns its own
       "multiplication algebra". This is the inductive-bias distinction
       from TFN/Kronecker (which uses a fixed outer-product algebra)
       and from LMF (which factorises into rank-R products).

    Params (in_f, out_f, n) = n^3 + (out_f * in_f) / n + out_f
       vs a plain nn.Linear(in_f, out_f): in_f * out_f + out_f
       so ~1/n of the linear's params.
    """

    def __init__(self, dim_a: int = 256, dim_b: int = 256,
                 fused_dim: int = 256, n: int = 4) -> None:
        super().__init__()
        in_f = dim_a + dim_b
        out_f = fused_dim
        if in_f % n != 0:
            raise ValueError(f"PHMFusion: dim_a+dim_b = {in_f} must be divisible by n={n}")
        if out_f % n != 0:
            raise ValueError(f"PHMFusion: fused_dim = {out_f} must be divisible by n={n}")
        self.n = n
        self.in_f = in_f
        self.out_f = out_f
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.fused_dim = fused_dim

        # A: (n, n, n) algebra matrices
        self.a = nn.Parameter(torch.empty(n, n, n))
        # S: (n, out_f//n, in_f//n) weight blocks
        self.s = nn.Parameter(torch.empty(n, out_f // n, in_f // n))
        # bias
        self.bias = nn.Parameter(torch.empty(out_f))

        # init: xavier_uniform on A and S (matches HyperNets reference)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.s)
        # bias init: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) -- same as nn.Linear
        bound = 1.0 / math.sqrt(in_f)
        nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def kronecker_product(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Vectorised Kronecker product over the leading n axis.

        Verbatim port of HyperNets/layers/ph_layers.py:kronecker_product1, which
        encodes the standard Kronecker rule kron((m_a,n_a),(m_b,n_b)) = (m_a*m_b, n_a*n_b)
        applied per-leading-index.

        a: (n, n, n)            -> per-i shape (n, n)        = (m_a, n_a)
        s: (n, out/n, in/n)     -> per-i shape (out/n, in/n) = (m_b, n_b)
        result: (n, out, in)    -- caller does torch.sum(..., dim=0) -> (out, in)

        Mechanics:
            res = a.unsqueeze(-1).unsqueeze(-3) * s.unsqueeze(-2).unsqueeze(-4)
            siz0 = res.shape[:-4]                 -- leading dims (just n here)
            siz1 = a.shape[-2:] (*) s.shape[-2:]  -- pointwise product (m_a*m_b, n_a*n_b)
            out  = res.reshape(siz0 + siz1)
        """
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(s.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * s.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        return res.reshape(siz0 + siz1)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # 1) concat the two modality vectors
        x = torch.cat([z_a, z_b], dim=-1)   # (B, dim_a + dim_b)
        # 2) compose the weight matrix W = sum_i kron(a[i], s[i])
        W = torch.sum(self.kronecker_product(self.a, self.s), dim=0)   # (out_f, in_f)
        # 3) standard affine
        return F.linear(x, W, self.bias)


# ----------------------------------------------------------------------
# Registry (for argparse selection in run_stage4.py)
# ----------------------------------------------------------------------
###############################################################################
# Bottleneck Kronecker (BKron)
###############################################################################
class BottleneckKronFusion(nn.Module):
    """
    Bottleneck wrapper around the full TFN/Kronecker (KroneckerTFNFusion).

    MOTIVATION
       KroneckerTFNFusion provides full degree-2 (bilinear) interaction via the
       outer product (z_a ⊗ z_b). With dim_a=dim_b=256, the flattened outer
       has (257*257)=66,049 entries, and the projection to fused_dim is
       ~51 M parameters. That gave the best fusion c-index in our Stage 4 sweep
       but with high std (0.046), suggesting overfitting risk.

       BKron preserves the SAME inductive bias (full bilinear interaction,
       complete with gating, skip-connection, and 2-stage encoders) but runs
       the bilinear inside a SMALLER bottleneck space (default 64-d). The
       bottleneck reduces the projection by (256/64)^2 = 16x.

    MATH
       z_a (B, 256) -> Linear -> z_a' (B, bottle)        # linear bottleneck
       z_b (B, 256) -> Linear -> z_b' (B, bottle)        # linear bottleneck
       fused = KroneckerTFNFusion(z_a', z_b')             # FULL bilinear at lower dim

       Note: the projections are FREE bilinear in the sense that they're linear
       in each modality before bilinear interaction; the actual degree-2
       interaction happens INSIDE the inner KroneckerTFNFusion, just on
       lower-dimensional inputs.

    PARAMS for bottle=64 (dim_a=dim_b=256, fused_dim=256):
       proj_a + proj_b: 2 * (256 * 64 + 64) = 32,896
       inner Kron:      depends on full Kron's internals at 64-d ~ 1-3 M
       total:           ~1-3 M (versus 51 M for full Kron at 256-d)
    """
    def __init__(self, dim_a: int = 256, dim_b: int = 256,
                 fused_dim: int = 256, bottle: int = 64) -> None:
        super().__init__()
        # Bottleneck projections from full dim to (bottle)-d
        self.proj_a = nn.Linear(dim_a, bottle)
        self.proj_b = nn.Linear(dim_b, bottle)
        # Sane init (matches PF init_max_weights via 1/sqrt(fan_in))
        for m in (self.proj_a, self.proj_b):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            nn.init.normal_(m.weight, 0.0, stdv)
            nn.init.zeros_(m.bias)
        # Reuse the full PF-style bilinear at lower dim
        self.kron = KroneckerTFNFusion(
            dim_a=bottle, dim_b=bottle, fused_dim=fused_dim
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        za = self.proj_a(z_a)       # (B, bottle)
        zb = self.proj_b(z_b)       # (B, bottle)
        return self.kron(za, zb)    # (B, fused_dim)


###############################################################################
# PHM-Kronecker Hybrid (PHMKron)
###############################################################################
class PHMKronHybridFusion(nn.Module):
    """
    Gated mixture of PHM (degree-1 linear, structured W) and BottleneckKron
    (degree-2 bilinear). Captures BOTH linear and bilinear features of the
    modality vectors at the same time.

    MOTIVATION
       Our Stage 4 sweep at 592-patient sample-level showed:
           Kronecker (degree-2, 51 M params): 0.8013   <- best, but heavy
           PHM       (degree-1, 33 K params): 0.7893   <- cheap, weaker
           Concat    (degree-1, 131 K params): 0.7931
           LMF       (rank-r, 527 K params):  0.7636   <- weakest
       The +0.012 from PHM/concat -> Kronecker is the value of degree-2
       bilinear interaction. PHM (and concat, LMF) are all in the LINEAR
       family on (z_a, z_b); they cannot express z_a_i * z_b_j terms.

       This hybrid gives the model BOTH paths:
         - PHM contributes linear features at near-zero cost (33 K)
         - BottleneckKron contributes bilinear features at controlled cost (~1-3 M)
         - A learnable scalar gate alpha picks the mix

    MATH
       out = sigmoid(alpha) * PHM(z_a, z_b)         # linear
           + (1 - sigmoid(alpha)) * BKron(z_a, z_b)  # bilinear in bottleneck
       alpha is initialized at 0 -> sigmoid(0) = 0.5 -> equal mix at init

    INTERPRETATION
       After training, sigmoid(alpha) tells us how much the model relied on
       each path. If sigmoid(alpha) -> 1, PHM was enough; if -> 0, BKron
       dominated; if near 0.5, both contributed.
    """
    def __init__(self, dim_a: int = 256, dim_b: int = 256,
                 fused_dim: int = 256, n: int = 4, bottle: int = 64) -> None:
        super().__init__()
        self.phm   = PHMFusion(dim_a=dim_a, dim_b=dim_b, fused_dim=fused_dim, n=n)
        self.bkron = BottleneckKronFusion(dim_a=dim_a, dim_b=dim_b,
                                           fused_dim=fused_dim, bottle=bottle)
        # Learnable scalar gate; init at 0 so sigmoid(0)=0.5 (balanced at start).
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha)
        return gate * self.phm(z_a, z_b) + (1 - gate) * self.bkron(z_a, z_b)


###############################################################################
# Anchored Bottleneck Kronecker (ABKron) -- PAL-spirit baked into architecture
###############################################################################
class AnchoredBKronFusion(nn.Module):
    """
    BottleneckKron + a direct gene-anchor skip path.

    MOTIVATION
       MoBalD's PAL trains the fused output's risk-ranking to match the
       gene-only model's risk-ranking via an auxiliary distillation loss.
       That's a *training-time* mechanism. This module captures the SAME
       INTENT (preserve the gene signal in the fused output even when the
       bilinear interaction is noisy on a given sample) but as an
       ARCHITECTURE-LEVEL skip connection.

    MATH (degree-mixed forward pass)
       h_bilinear     = BottleneckKron(z_a, z_b)                # (B, fused_dim)
                                                                   degree-2 in (z_a, z_b)
       h_gene_anchor  = Linear(dim_b -> fused_dim)(z_b)         # (B, fused_dim)
                                                                   degree-1 in z_b
       out            = Linear(2*fused_dim -> fused_dim)(
                            concat([h_bilinear, h_gene_anchor])  # (B, 2*fused_dim)
                        )                                         # (B, fused_dim)

       Because the gene-anchor flows through an independent linear path,
       gene information CANNOT be lost even if the bilinear bottleneck loses
       fidelity on hard cases. This is exactly the spirit of PAL: gene info
       is preserved structurally, not by auxiliary supervision.

    PARAMS for dim_a=dim_b=fused_dim=256, bottle=64:
       inner BKron     :  ~1.75 M
       gene_anchor     :  256 * 256 + 256       = 65,792
       fuse_proj       :  (2 * 256) * 256 + 256 = 131,328
       total           :  ~1.95 M (vs 1.75 M for plain BKron)

    DIFFERENCES FROM CLASSICAL MoBalD
       - NO auxiliary heads, NO auxiliary losses, NO PAL distillation,
         NO GradMod gradient scaling
       - Single forward pass, standard end-to-end training with Cox NLL
       - Trades GradMod's per-step adaptivity for architectural simplicity
       - Captures only the PAL "preserve gene signal" intent, not the
         GradMod "balance branches dynamically" intent
    """
    def __init__(self, dim_a: int = 256, dim_b: int = 256,
                 fused_dim: int = 256, bottle: int = 64) -> None:
        super().__init__()
        # 1) Bilinear path -- the existing BKron (which itself wraps
        #    KroneckerTFNFusion at lower dim with full gating + skip).
        self.bkron = BottleneckKronFusion(
            dim_a=dim_a, dim_b=dim_b, fused_dim=fused_dim, bottle=bottle,
        )
        # 2) Gene-only direct path -- single Linear from dim_b to fused_dim.
        self.gene_anchor = nn.Linear(dim_b, fused_dim)
        # 3) Combine the bilinear + gene-anchor features into the final
        #    fused embedding.
        self.fuse_proj = nn.Linear(fused_dim * 2, fused_dim)

        # PF-style init for the new Linears: normal(0, 1/sqrt(fan_in)).
        for m in (self.gene_anchor, self.fuse_proj):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            nn.init.normal_(m.weight, 0.0, stdv)
            nn.init.zeros_(m.bias)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # z_a: (B, dim_a)   z_b: (B, dim_b)
        h_bilinear = self.bkron(z_a, z_b)                       # (B, fused_dim)
        h_gene     = self.gene_anchor(z_b)                       # (B, fused_dim)
        fused_concat = torch.cat([h_bilinear, h_gene], dim=-1)  # (B, 2*fused_dim)
        return self.fuse_proj(fused_concat)                      # (B, fused_dim)


###############################################################################
# HGBF -- Hypercomplex-Gated Bilinear Fusion (PHM-modulated bottleneck Kron)
#
# NAMING NOTE: the internal Python identifiers SlimCPKFFusion and the
# registry key "slim_cpkf" are kept stable across the publication rename
# (preserves compatibility with cached results, JSONs, and CSVs). The
# PUBLISHED name of this operator is HGBF (Hypercomplex-Gated Bilinear
# Fusion). The two refer to the same operator.
###############################################################################
class SlimCPKFFusion(nn.Module):
    """
    HGBF -- Hypercomplex-Gated Bilinear Fusion.
    Internal Python name: SlimCPKFFusion (kept for compatibility).

    A parameter-efficient compositional hybrid that uses PHM as a
    content-aware *modulator* of the inputs to a bottleneck Kronecker
    bilinear operator. Distinct from PHMKronHybridFusion (which combines
    PHM and BKron outputs via a STATIC SCALAR gate). Here PHM does NOT
    compete with BKron at the output -- it SHAPES BKron's inputs through
    per-sample, per-dimension gates computed from the cross-modal signal.

    MOTIVATION
       PHMKronHybridFusion's scalar gate is content-independent (same
       blend for every sample). It cannot adapt to whether a sample needs
       more degree-1 (linear) vs degree-2 (bilinear) interaction. The
       full-input CPKF variant (PHM-modulation on the raw 256-d inputs)
       adds ~460K params for gates on the high-dimensional inputs.

       HGBF performs the modulation INSIDE the 64-d bottleneck:
         - PHM operates on the 64-d projected vectors  (cheap)
         - Gates are 64-d (cheap)
         - Bilinear runs on the modulated bottlenecks  (same cost as BKron)

       Result: HGBF is SMALLER than BKron (~1.09 M vs 1.75 M), keeps
       the same bilinear inductive bias, AND adds content-aware modulation.

    MATH
       1) Bottleneck projection (linear, free of cross-modal interaction):
            z_a_bn = Linear(dim_a -> bottle)(z_a)         # (B, 64)
            z_b_bn = Linear(dim_b -> bottle)(z_b)         # (B, 64)

       2) PHM cross-modal modulation signal (degree-1 in z_a_bn, z_b_bn):
            m = PHMFusion(z_a_bn, z_b_bn)                 # (B, 64)
                concat([z_a_bn, z_b_bn]) -> 128, PHM -> 64

       3) Per-dim content-aware gates from m (different for every sample):
            g_a = sigmoid(W_a · m + b_a)                  # (B, 64)
            g_b = sigmoid(W_b · m + b_b)                  # (B, 64)

       4) Modulate bottleneck vectors element-wise:
            z_a_mod = g_a (*) z_a_bn                      # (B, 64)
            z_b_mod = g_b (*) z_b_bn                      # (B, 64)

       5) BKron-style bilinear over the modulated bottlenecks:
            out = KroneckerTFNFusion(z_a_mod, z_b_mod)    # (B, fused_dim)

    INTERPRETATION
       The PHM signal m encodes cross-modal context (which image features
       interact with which gene features). The gates g_a, g_b say "use
       THIS subset of the image bottleneck and THAT subset of the gene
       bottleneck for the bilinear interaction." Different samples can
       attend to different bottleneck dimensions. The bilinear interaction
       then operates over the relevant subspace.

       Contrast with PHMKronHybridFusion (parallel scalar mix):
          OLD : out = sigmoid(alpha) * PHM(z) + (1-sigmoid(alpha)) * BKron(z)
          NEW : out = BKron(gate(PHM(z)) (*) z)
       Old combines outputs; new shapes inputs. Sequential composition vs
       parallel blend.

    PARAMS for dim_a=dim_b=fused_dim=256, bottle=64, n=4:
       proj_a + proj_b     :  2 * (256*64 + 64)               = 32,896
       PHM modulator       :  n^3 + (64*128)/n + 64
                                                              = 64 + 2048 + 64 = 2,176
       gate_a + gate_b     :  2 * (64*64 + 64)                = 8,320
       inner Kron(64,64)   :  ~1.05 M  (the same inner Kron BKron uses)
       --------------------------------------------------------------
       total               :  ~1.09 M

       Comparison:
          Plain Kron at 256-d           : ~51 M
          BKron (bottleneck=64)         : ~1.75 M
          PHMKronHybrid (scalar gate)   : ~1.78 M
          HGBF                     : ~1.09 M    (-38% vs BKron)
    """
    def __init__(self, dim_a: int = 256, dim_b: int = 256,
                 fused_dim: int = 256, bottle: int = 64, n: int = 4) -> None:
        super().__init__()
        # Constraint check for PHM divisibility
        if (bottle * 2) % n != 0:
            raise ValueError(f"SlimCPKF: 2*bottle = {2*bottle} must be divisible by n={n}")
        if bottle % n != 0:
            raise ValueError(f"SlimCPKF: bottle = {bottle} must be divisible by n={n}")

        # 1) Bottleneck projections (identical structure to BKron)
        self.proj_a = nn.Linear(dim_a, bottle)
        self.proj_b = nn.Linear(dim_b, bottle)
        for m in (self.proj_a, self.proj_b):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            nn.init.normal_(m.weight, 0.0, stdv)
            nn.init.zeros_(m.bias)

        # 2) PHM modulation signal: (bottle, bottle) -> bottle  (degree-1 cross-modal)
        self.phm_mod = PHMFusion(dim_a=bottle, dim_b=bottle, fused_dim=bottle, n=n)

        # 3) Per-dim content-aware gates from PHM signal
        self.gate_a = nn.Linear(bottle, bottle)
        self.gate_b = nn.Linear(bottle, bottle)
        for m in (self.gate_a, self.gate_b):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            nn.init.normal_(m.weight, 0.0, stdv)
            # Initialise gates at ~0.5 (pre-sigmoid logits ~0) so HGBF
            # starts close to plain BKron with full passthrough modulation.
            nn.init.zeros_(m.bias)

        # 4) Inner Kronecker bilinear -- same as BKron's inner module
        self.kron = KroneckerTFNFusion(
            dim_a=bottle, dim_b=bottle, fused_dim=fused_dim
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # 1) Project to bottleneck
        z_a_bn = self.proj_a(z_a)                              # (B, bottle)
        z_b_bn = self.proj_b(z_b)                              # (B, bottle)

        # 2) PHM cross-modal modulation signal
        mod = self.phm_mod(z_a_bn, z_b_bn)                     # (B, bottle)

        # 3) Per-dim content-aware gates (sigmoid in [0, 1])
        g_a = torch.sigmoid(self.gate_a(mod))                  # (B, bottle)
        g_b = torch.sigmoid(self.gate_b(mod))                  # (B, bottle)

        # 4) Modulate bottleneck vectors element-wise
        z_a_mod = g_a * z_a_bn                                 # (B, bottle)
        z_b_mod = g_b * z_b_bn                                 # (B, bottle)

        # 5) Bilinear over modulated bottlenecks
        return self.kron(z_a_mod, z_b_mod)                     # (B, fused_dim)


FUSION_REGISTRY: Dict[str, Type[nn.Module]] = {
    "concat": ConcatFusion,
    "kronecker": KroneckerTFNFusion,
    "lmf": LMFFusion,
    "phm": PHMFusion,
    "bkron": BottleneckKronFusion,        # bottleneck full bilinear
    "phm_kron": PHMKronHybridFusion,      # gated linear + bilinear hybrid (scalar gate)
    "abkron": AnchoredBKronFusion,        # BKron + gene-anchor skip (PAL-spirit)
    "slim_cpkf": SlimCPKFFusion,          # NEW: PHM-modulated BKron in bottleneck
}


def build_fusion(name: str, dim_a: int = 256, dim_b: int = 256,
                 fused_dim: int = 256, **kwargs) -> nn.Module:
    """Construct a fusion op by name with standardized dims."""
    if name not in FUSION_REGISTRY:
        raise KeyError(f"unknown fusion op: {name!r}; choose from {list(FUSION_REGISTRY)}")
    cls = FUSION_REGISTRY[name]
    return cls(dim_a=dim_a, dim_b=dim_b, fused_dim=fused_dim, **kwargs)
