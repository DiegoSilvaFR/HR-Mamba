import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Pure PyTorch Mamba-like selective SSM block.

    This is not the official fused Mamba implementation.
    It is a simplified selective state-space layer:

        h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        y_t = C_t * h_t + D * x_t

    where delta_t, B_t, and C_t are input-dependent.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # Produces input-dependent delta, B, and C
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state)

        # Stable diagonal state matrix A.
        # A is forced to be negative.
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).repeat(self.d_inner, 1)
        )

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns: (B, L, D)
        """

        batch_size, seq_len, _ = x.shape

        u, z = self.in_proj(x).chunk(2, dim=-1)  # (B, L, d_inner), (B, L, d_inner)

        # Local mixing, similar in spirit to Mamba's depthwise conv
        u = u.transpose(1, 2)                    # (B, d_inner, L)
        u = self.conv1d(u)[..., :seq_len]        # crop to original length
        u = F.silu(u).transpose(1, 2)            # (B, L, d_inner)

        params = self.x_proj(u)

        delta, B_t, C_t = torch.split(
            params,
            [self.d_inner, self.d_state, self.d_state],
            dim=-1,
        )

        delta = F.softplus(delta + self.dt_bias)  # (B, L, d_inner)

        # Negative diagonal A for stable dynamics
        A = -torch.exp(self.A_log.float()).to(dtype=x.dtype)  # (d_inner, d_state)

        h = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=x.dtype,
        )

        ys = []

        for t in range(seq_len):
            dt = delta[:, t, :]      # (B, d_inner)
            u_t = u[:, t, :]         # (B, d_inner)
            B_vec = B_t[:, t, :]     # (B, d_state)
            C_vec = C_t[:, t, :]     # (B, d_state)

            A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt.unsqueeze(-1) * B_vec.unsqueeze(1)

            h = A_bar * h + B_bar * u_t.unsqueeze(-1)

            y_t = (h * C_vec.unsqueeze(1)).sum(dim=-1) + self.D.to(dtype=x.dtype) * u_t
            ys.append(y_t)

        y = torch.stack(ys, dim=1)   # (B, L, d_inner)

        # Gate
        y = y * F.silu(z)

        y = self.out_proj(y)
        return self.dropout(y)


class BiMambaLikeBlock(nn.Module):
    """
    Bidirectional Mamba-like block.

    Processes the sequence forward and backward, then combines both directions.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.forward_ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dropout=dropout,
        )

        self.backward_ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dropout=dropout,
        )

        self.mix = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x_norm = self.norm(x)

        y_forward = self.forward_ssm(x_norm)

        y_backward = torch.flip(
            self.backward_ssm(torch.flip(x_norm, dims=[1])),
            dims=[1],
        )

        y = torch.cat([y_forward, y_backward], dim=-1)
        y = self.mix(y)

        return residual + self.dropout(y)


class HRMambaLikeRegressor(nn.Module):
    """
    Mamba-like backbone for heart-rate regression from accelerometer windows.

    Expected input:
        x: (B, 300, 3)

    Output:
        predicted HR: (B, 1)
    """

    def __init__(
        self,
        input_channels: int = 3,
        seq_len: int = 300,
        patch_size: int = 15,
        stride: int = 15,
        d_model: int = 128,
        depth: int = 4,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride

        self.patch_embed = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride,
        )

        n_tokens = (seq_len - patch_size) // stride + 1

        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d_model))

        self.blocks = nn.ModuleList(
            [
                BiMambaLikeBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    d_conv=d_conv,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
            x: (B, T, C)
            or
            x: (B, 1, T, C)

        Returns:
            HR prediction: (B, 1)
        """

        if x.ndim == 4:
            x = x.squeeze(1)      # (B, T, C)

        x = x.transpose(1, 2)     # (B, C, T)

        x = self.patch_embed(x)   # (B, D, L)
        x = x.transpose(1, 2)     # (B, L, D)

        x = x + self.pos_embed[:, : x.shape[1], :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # For HR regression, mean pooling is safer than CLS at first
        x = x.mean(dim=1)

        return self.head(x)