import torch


class FlowMatchScheduler:
    def __init__(
        self,
        shift: float,
        sigma_min: float,
        sigma_max: float,
        num_train_timesteps: int,
        extra_one_step: bool,
    ):
        self.num_train_timesteps = num_train_timesteps
        n = num_train_timesteps
        if extra_one_step:
            self.sigmas = torch.linspace(sigma_max, sigma_min, n + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_max, sigma_min, n)
        self.sigmas = shift * self.sigmas / (1 + (shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * n
        # Bell-shape weights for training_weight() (centered around the middle timestep,
        # normalised so that the weights sum to n).
        x = self.timesteps
        y = torch.exp(-2 * ((x - n / 2) / n) ** 2)
        y_shifted = y - y.min()
        self.linear_timesteps_weights = y_shifted * (n / y_shifted.sum())

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B*T, C, H, W]
            - noise: the noise with shape [B*T, C, H, W]
            - timestep: the timestep with shape [B*T]
        Output: the corrupted latent with shape [B*T, C, H, W]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        """
        Input:
            - timestep: the timestep with shape [B*T]
        Output: the corresponding weighting [B*T]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0
        )
        return self.linear_timesteps_weights[timestep_id]
