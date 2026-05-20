from torch.utils.data import Dataset
import torch
import numpy as np

def beta_scaled(a, b, alpha, beta, size, device='cpu'):
    beta_sample = torch.distributions.Beta(alpha, beta).sample(size).to(device)
    return a + (b - a) * beta_sample

class SlabWaveguideDataset(Dataset):
    def __init__(self, n_dataset=1024, stochastic=False, dx=1/20, x_range=(-5, 5), device='cpu'):
        self.device = device
        self.dx = dx
        self.x_range = x_range
        self.n_dataset = n_dataset
        self.x, self.Nx = self.generate_grid()
        self.x = self.x.to(self.device)
        self.stochastic = stochastic
        if not stochastic:
            self.generate_dataset_features()

    def __len__(self):
        return self.n_dataset

    def generate_grid(self):
        left_length_x, right_length_x = self.x_range
        xr = torch.arange(0, right_length_x, self.dx)
        xl = torch.arange(0, -left_length_x, self.dx)
        x = torch.cat((-torch.flip(xl, (0,))[:-1], xr))
        Nx = len(x)
        return x, Nx

    def generate_sample_features(self):
        width = torch.rand(1).item() + 1
        n_clad = 1 + torch.rand(1).item()
        n_sub = 1 + torch.rand(1).item()
        n_core = beta_scaled(max(n_clad, n_sub) + 0.05, 3.5, 0.5, 0.5, (1,), 'cpu').item()

        TE = 1.0 * (torch.rand(1) < 0.5)

        lambd = 1 + torch.rand(1)
        wavenumber = 2 * np.pi / lambd

        features = [
            width,
            n_clad,
            n_sub,
            n_core,
            TE,
            wavenumber
        ]
        return features

    def generate_dataset_features(self):
        self.dataset_features = []
        for _ in range(self.n_dataset):
            self.dataset_features.append(self.generate_sample_features())

    def __getitem__(self, idx):
        if not self.stochastic:
            profile, features = self.build_profile(*self.dataset_features[idx])
        else:
            profile, features = self.build_profile(*self.generate_sample_features())
        profile = torch.cat((self.x.unsqueeze(0), profile.to(self.device)), dim=0)
        return profile, features.to(self.device)

    def build_profile(self, width, n_clad, n_sub, n_core, TE, wavenumber):
        profile = torch.empty((3, self.Nx), device=self.device)

        sub_mask = self.x < -width / 2
        core_mask = torch.abs(self.x) <= width / 2

        profile[0] = n_clad ** 2
        profile[0][sub_mask] = n_sub ** 2
        profile[0][core_mask] = n_core ** 2

        profile[1] = TE
        profile[2] = wavenumber ** 2

        features = torch.tensor([
            wavenumber,
            width,
            n_clad ** 2,
            n_sub ** 2,
            n_core ** 2,
            TE
        ])

        return profile, features

    def parameter_sweep(self, param_dict_list):
        sweep_devices = []

        for params in param_dict_list:
            width = params.get('width')
            n_clad = params.get('n_clad')
            n_sub = params.get('n_sub')
            n_core = params.get('n_core')
            TE = params.get('TE')
            lambd = params.get('wavelength')
            wavenumber = 2 * np.pi / lambd

            profile, features = self.build_profile(width, n_clad, n_sub, n_core, TE, wavenumber)
            profile = torch.cat((self.x.unsqueeze(0), profile), dim=0)
            sweep_devices.append((profile, features))

        return sweep_devices


class RibWaveguideDataset(Dataset):
    def __init__(self, n_dataset=32, stochastic = False, dx=1/20, dy=1/20, x_range=(-5.0, 5.0), y_range=(-5, 5), device='cpu'):
        self.n_dataset = n_dataset
        self.device = device

        self.dx = dx
        self.dy = dy
        self.x_range = x_range
        self.y_range = y_range

        # Generate the coordinate grid once and store it on the target device
        self.xy, self.Nx, self.Ny = self.generate_grid()
        self.xy = self.xy.to(self.device)

        self.stochastic = stochastic
        if not stochastic:
            self.generate_dataset_features()


    def __len__(self):
        return self.n_dataset

    def generate_grid(self):
        """Generates the 2D coordinate grid using torch."""
        left_length_x, right_length_x = self.x_range
        lower_length_y, upper_length_y = self.y_range

        xr = torch.arange(0, right_length_x, self.dx).reshape(-1, 1)
        xl = torch.arange(0, -left_length_x, self.dx).reshape(-1, 1)
        x = torch.cat((-torch.flip(xl, (0,))[:-1], xr)).squeeze()

        yu = torch.arange(0, upper_length_y, self.dy).reshape(-1, 1)
        yl = torch.arange(0, -lower_length_y, self.dy).reshape(-1, 1)
        y = torch.cat((-torch.flip(yl, (0,))[:-1], yu)).squeeze()

        Nx, Ny = len(x), len(y)
        xx, yy = torch.meshgrid(x, y, indexing='xy')  # (Ny, Nx)
        xy = torch.stack([xx, yy], dim=-1)             # shape: (Ny, Nx, 2)
        return xy, Nx, Ny

    def __getitem__(self, idx):
        if not self.stochastic:
            profiles, features = self.build_profile(*self.dataset_features[idx])
        else:
            profiles, features = self.build_profile(*self.generate_sample_features())
        input_tensor = torch.cat((self.xy, profiles), dim=-1)
        return input_tensor.permute(2, 1, 0), features

    def generate_sample_features(self):
        wavelength = torch.rand(1, device=self.device) + 1.0           # Range: [1.0, 2.0]
        wavenumber = (2 * torch.pi) / wavelength                        # Free-space wavenumber k0

        core_width = torch.rand(1, device=self.device) * 2.0 + 2.0      # Range: [2.0, 4.0]
        while True:
            core_height = beta_scaled(1e-1, 2e0, 0.5, 0.5, (1,), device=self.device)       # Range: [0.1, 2.0]
            rib_height = beta_scaled(1e-1, 2e0, 0.5, 0.5, (1,), device=self.device)       # Range: [0.1, 2.0]
            if core_height + rib_height >= 1.0:
                break

        n_clad = torch.rand(1, device=self.device) * 1.0 + 1.0          # Range: [1.0, 2.0]
        n_sub = torch.rand(1, device=self.device) * 1 + 2.0           # Range: [2.0, 3.0]
        n_core = beta_scaled(n_sub + 5e-2, 35e-1, 0.5, 0.5, (1,), device=self.device)       # Range: [n_sub + 0.05, 3.5]
        TE = (torch.rand(1, device=self.device) > 0.5).float()          # 1.0 for TE, 0.0 for TM

        features = [
            wavenumber,
            core_width,
            core_height,
            rib_height,
            n_clad,
            n_sub,
            n_core,
            TE
        ]
        return features

    def generate_dataset_features(self):
        self.dataset_features = []
        for i in range(self.n_dataset):
            self.dataset_features.append(self.generate_sample_features())

    def build_profile(self, wavenumber, core_width, core_height, rib_height, n_clad, n_sub, n_core, TE):
        profile = torch.empty((self.Ny, self.Nx, 3), device=self.device)

        sub_mask = self.xy[:, :, 1] < -rib_height
        rib_mask = (self.xy[:, :, 1] >= -rib_height) & (self.xy[:, :, 1] <= 0)
        core_mask = (torch.abs(self.xy[:, :, 0]) <= core_width / 2) & \
                    (self.xy[:, :, 1] <= core_height) & \
                    (self.xy[:, :, 1] >= 0)

        # Channel 0: Refractive Index (n)
        profile[:, :, 0] = n_clad**2
        profile[:, :, 0][sub_mask] = n_sub**2
        profile[:, :, 0][rib_mask] = n_core**2
        profile[:, :, 0][core_mask] = n_core**2
        # Channel 1: Polarization (TE)
        profile[:, :, 1] = TE
        # Channel 2: Wavenumber (k)
        profile[:, :, 2] = wavenumber**2

        features = torch.cat([
            wavenumber,
            core_width,
            core_height,
            rib_height,
            n_clad**2,
            n_sub**2,
            n_core**2,
            TE
        ])

        return profile, features

    def generate_from_specs(self, specs):
        """
        Generates a batch of devices based on a list of specification dictionaries.

        Args:
            specs (list of dict): Each dict defines a device spec.
                                  Possible keys: wavelength, core_width, core_height,
                                  rib_height, n_clad, n_sub, n_core, TE.

        Returns:
            input_batch: Tensor of shape (batch_size, channels, Nx, Ny)
            features_batch: Tensor of shape (batch_size, n_features)
        """
        input_batch = []
        features_batch = []

        for spec in specs:
            profile, features = self.generate_sample_from_spec(spec)

            # Concatenate grid (x,y) with profile (n,TE,k)
            input_tensor = torch.cat((self.xy, profile), dim=-1)

            # Permute to CNN format (channels, Nx, Ny)
            input_batch.append(input_tensor.permute(2, 1, 0))
            features_batch.append(features)

        return torch.stack(input_batch), torch.stack(features_batch)

    def generate_sample_from_spec(self, spec):
        wavelength = torch.tensor([spec.get("wavelength", float(torch.rand(1) + 1.0))], device=self.device)
        wavenumber = (2 * torch.pi) / wavelength

        core_width = torch.tensor([spec.get("core_width", float(torch.rand(1) * 2.0 + 2.0))], device=self.device)
        core_height = torch.tensor([spec.get("core_height", float(beta_scaled(1e-1, 2e0, 0.5, 0.5, (1,), device=self.device)))], device=self.device)
        rib_height = torch.tensor([spec.get("rib_height", float(beta_scaled(1e-1, 2e0, 0.5, 0.5, (1,), device=self.device)))], device=self.device)


        n_clad = torch.tensor([spec.get("n_clad", float(torch.rand(1) * 1.0 + 1.0))], device=self.device)
        n_sub = torch.tensor([spec.get("n_sub", float(torch.rand(1) * 1.5 + 2.0))], device=self.device)

        if "n_core" in spec:
            n_core = torch.tensor([spec["n_core"]], device=self.device)
        else:
            n_core = n_sub + beta_scaled(5e-2, 5e-1, 0.5, 2.0, (1,), device=self.device)

        TE = torch.tensor([spec.get("TE", float((torch.rand(1) > 0.5).float()))], device=self.device)

        profile = torch.empty((self.Ny, self.Nx, 3), device=self.device)

        sub_mask = self.xy[:, :, 1] < -rib_height
        rib_mask = (self.xy[:, :, 1] >= -rib_height) & (self.xy[:, :, 1] <= 0)
        core_mask = (torch.abs(self.xy[:, :, 0]) < core_width / 2) & \
                    (self.xy[:, :, 1] < core_height) & \
                    (self.xy[:, :, 1] > 0)

        profile[:, :, 0] = n_clad**2
        profile[:, :, 0][sub_mask] = n_sub**2
        profile[:, :, 0][rib_mask] = n_core**2
        profile[:, :, 0][core_mask] = n_core**2
        profile[:, :, 1] = TE
        profile[:, :, 2] = wavenumber**2

        features = torch.cat([
            wavenumber,
            core_width,
            core_height,
            rib_height,
            n_clad**2,
            n_sub**2,
            n_core**2,
            TE
        ])

        return profile, features

class ChannelDataset(Dataset):
    def __init__(self, n_dataset=32, stochastic=False, dx=1/20, dy=1/20, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0), device='cpu'):
        self.n_dataset = n_dataset
        self.device = device

        self.dx = dx
        self.dy = dy
        self.x_range = x_range
        self.y_range = y_range

        # Generate the coordinate grid once
        self.xy, self.Nx, self.Ny = self.generate_grid()
        self.xy = self.xy.to(self.device)

        self.stochastic = stochastic
        if not stochastic:
            self.generate_dataset_features()

    def __len__(self):
        return self.n_dataset

    def generate_grid(self):
        left_length_x, right_length_x = self.x_range
        lower_length_y, upper_length_y = self.y_range

        xr = torch.arange(0, right_length_x, self.dx).reshape(-1, 1)
        xl = torch.arange(0, -left_length_x, self.dx).reshape(-1, 1)
        x = torch.cat((-torch.flip(xl, (0,))[:-1], xr)).squeeze()

        yu = torch.arange(0, upper_length_y, self.dy).reshape(-1, 1)
        yl = torch.arange(0, -lower_length_y, self.dy).reshape(-1, 1)
        y = torch.cat((-torch.flip(yl, (0,))[:-1], yu)).squeeze()

        Nx, Ny = len(x), len(y)
        xx, yy = torch.meshgrid(x, y, indexing='xy')  # (Ny, Nx)
        xy = torch.stack([xx, yy], dim=-1)  # shape: (Ny, Nx, 2)
        return xy, Nx, Ny

    def __getitem__(self, idx):
        if not self.stochastic:
            profiles, features = self._build_profile(*self.dataset_features[idx])
        else:
            profiles, features = self._build_profile(*self.generate_sample_features())
        input_tensor = torch.cat((self.xy, profiles), dim=-1)
        return input_tensor.permute(2, 1, 0), features

    def generate_dataset_features(self):
        self.dataset_features = []
        for i in range(self.n_dataset):
            self.dataset_features.append(self.generate_sample_features())

    def _build_profile(self, wavenumber, core_width, core_height, n_sub, n_core, TE):
        profile = torch.empty((self.Ny, self.Nx, 3), device=self.device)
        core_mask = (torch.abs(self.xy[:, :, 0]) <= core_width / 2) & \
                    (torch.abs(self.xy[:, :, 1]) <= core_height / 2)

        # Channel 0: Refractive Index Squared (n^2)
        profile[:, :, 0] = n_sub ** 2
        profile[:, :, 0][core_mask] = n_core**2
        # Channel 1: Polarization
        profile[:, :, 1] = TE
        # Channel 2: Wavenumber (k^2)
        profile[:, :, 2] = wavenumber**2

        features = torch.cat([
            wavenumber,
            core_width,
            core_height,
            n_sub**2,
            n_core**2,
            TE
        ])

        return profile, features

    def generate_sample_features(self):
        wavelength = torch.rand(1, device=self.device) + 1.0
        wavenumber = (2 * torch.pi) / wavelength

        core_width = beta_scaled(1.0, 3.0, 0.75, 0.75, (1,), device=self.device)
        core_height = beta_scaled(1.0, 3.0, 0.75, 0.75, (1,), device=self.device)

        n_sub = torch.rand(1, device=self.device) * 2 + 1.0
        n_core = beta_scaled(n_sub + 0.05, 3.5, 0.5, 0.5, (1,), device=self.device)

        TE = (torch.rand(1, device=self.device) > 0.5).float()

        return [wavenumber, core_width, core_height, n_sub, n_core, TE]

    def generate_samples_from_specs(self, specs_list):
        """Generates a batch of samples from specific user-provided parameters."""
        samples = []

        for specs in specs_list:

            wavelength = torch.tensor([specs['wavelength']], device=self.device, dtype=torch.float32)
            wavenumber = (2 * torch.pi) / wavelength
            core_width = torch.tensor([specs['core_width']], device=self.device, dtype=torch.float32)
            core_height = torch.tensor([specs['core_height']], device=self.device, dtype=torch.float32)
            n_sub = torch.tensor([specs['n_sub']], device=self.device, dtype=torch.float32)
            n_core = torch.tensor([specs['n_core']], device=self.device, dtype=torch.float32)
            TE = torch.tensor([specs['TE']], device=self.device, dtype=torch.float32)

            profile, features = self._build_profile(
                wavenumber, core_width, core_height, n_sub, n_core, TE
            )
            input_tensor = torch.cat((self.xy, profile), dim=-1)
            input_tensor = input_tensor.permute(2, 1, 0)

            samples.append((input_tensor, features))

        return samples