import torch.nn as nn
import torch

# -------- Spectral Convolution Layer --------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, F)
        F = x_ft.shape[-1]
        m = min(self.modes, F)

        x_ft_m = x_ft[:, :, :m]  # (B, C_in, m)
        w_real = self.weights_real[:, :, :m]  # (C_in, C_out, m)
        w_imag = self.weights_imag[:, :, :m]
        w = torch.complex(w_real, w_imag)  # (C_in, C_out, m)

        out_ft_m = torch.einsum("bcm,com->bom", x_ft_m, w)  # (B, C_out, m)

        out_ft = torch.zeros(B, self.out_channels, F, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :m] = out_ft_m

        x_out = torch.fft.irfft(out_ft, n=N, dim=-1)  # (B, C_out, N)
        return x_out


# -------- Fourier Neural Operator --------
class FNO1d(nn.Module):
    def __init__(self, in_dim, out_dim, width=64, modes=16, depth=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_dim, width, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1, padding=0)
        )

        self.spectral_blocks = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()

        for _ in range(depth):
            self.spectral_blocks.append(SpectralConv1d(width, width, modes))
            self.pointwise_convs.append(nn.Conv1d(width, width, kernel_size=1))

        self.output_proj = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv1d(width, out_dim, kernel_size=1, padding=0)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.input_proj(x)  # → (B, width, N)

        for spec, pw in zip(self.spectral_blocks, self.pointwise_convs):
            x = spec(x) + pw(x)
            x = self.activation(x)

        return self.output_proj(x)  # → (B, N, out_dim)

class PINO1D(nn.Module):
    def __init__(self, in_dim, out_dim, width=64, modes=32, depth=2):
        super().__init__()
        self.backbone = FNO1d(in_dim, width, width, modes, depth)

        self.field_head = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv1d(width, out_dim, kernel_size=1, padding=0),
        )

        self.eigenvalue = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        field = self.field_head(x)
        eigenvalue = self.eigenvalue(x)
        return field, eigenvalue


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    def compl_mul2d(self, input_ft, weights):
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.to(torch.float32)

        x_ft = torch.fft.rfft2(x, norm="ortho")

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        weights = torch.complex(
            self.weights_real[:, :, :m1, :m2],
            self.weights_imag[:, :, :m1, :m2]
        )
        weights_conj_flip = weights.flip(2).conj()

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, self.out_channels,  H, W//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights_conj_flip)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x


# ===== FNO Block (Spectral + Pointwise) =====
class FNOBlock2D(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.spectral(x) + self.pointwise(x))

class PINO2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, width=64, modes1=32, modes2=32, depth=3):
        super().__init__()

        self.fc0 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1)
        )

        self.blocks = nn.Sequential(*[
            FNOBlock2D(width, modes1, modes2) for _ in range(depth)
        ])

        self.fc1 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1)
        )

        self.field = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1)
        )

        self.eigenvalue = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(self.blocks(self.fc0(x)))
        return self.field(x), self.eigenvalue(x)