
import torch
import torch.nn as nn
from typing import Optional, Tuple
from utils import unnormalize_to_zero_to_one
import torch.nn.functional as F

class FlowModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        ode_solver: str = "euler",
    ):
        super().__init__()
        self.model = model
        self.channels = getattr(self.model, "channels", 3)
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else timesteps
        self.ode_solver = ode_solver.lower()

    def _prepare_t(self, t: torch.Tensor) -> torch.Tensor:
        if t.dtype == torch.long:
            return t
        return (t.clamp(0, 1) * (self.timesteps - 1)).round().long()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ##################################################################
        # TODO 4.1: Implement the forward pass of the flow matching model.
        # First, prepare the time step t,
        # then pass x_t and the prepared t to the model.
        ##################################################################
        # t_idx shape should be [B], where B is the batch size, dtype long
        # x_t shape is [B, C, H, W]
        # t_idx is the time step indices corresponding to t
        t_idx = self._prepare_t(t).to(x_t.device) # prepare time steps and pass to same device as x_t
        return self.model(x_t, t_idx)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


        

    @torch.no_grad()
    def ode_euler_step(self, x: torch.Tensor, t: torch.Tensor, dt: float) -> torch.Tensor:
        ##################################################################
        # TODO 4.1: Implement one Euler step of the ODE solver
        ##################################################################
        velocity = self.forward(x, t)  # v_theta(x, t)
        x_next = x + velocity * dt     # x(t + dt) = x(t) + v_theta(x(t), t) * dt
        return x_next
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def _time_grid(self, steps: int):
        """
        Produce a uniform grid in [0,1]
        """
        return torch.linspace(0.0, 1.0, steps=steps, device=self.device)

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int, int], solver: Optional[str] = None, steps: Optional[int] = None):
        """
        Sample images by integrating the learned ODE:
          dx/dt = v_theta(x, t)
        starting from x(0) ~ N(0,I) and integrating t in [0,1].
        """
        solver = (solver or self.ode_solver).lower()
        steps = steps or self.sampling_timesteps
        assert steps >= 2, "Number of sampling steps must be at least 2 for Euler integration"

        ##################################################################
        # TODO 4.1: Implement time grid and time step size,
        ##################################################################
        B, C, H, W = shape # For my own understanding
        x = torch.randn(shape, device=self.device)   # x(0) ~ N(0, I)

        # Uniform time grid in [0,1]
        ts = self._time_grid(steps)  # shape [steps]
        # dt = ts[1] - ts[0]           # scalar time step size
        dt = float(1.0 / (steps - 1))               # scalar step size

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

        # advance along the grid
        for i in range(steps - 1):
            t_curr = ts[i].expand(B)     # shape [B], required by UNet interface
            if solver == "euler":
                x = self.ode_euler_step(x, t_curr, dt)
            else:
                raise ValueError(f"Unknown ODE solver: {solver}")

        # final clamp to [0,1] for image writing convenience
        img = unnormalize_to_zero_to_one(x)
        return img

    @torch.no_grad()
    def sample_given_z(self, z: torch.Tensor, shape: Tuple[int, int, int, int], solver: Optional[str] = None, steps: Optional[int] = None):
        """
        Same as sample(), but you provide the starting noise 'z' already
        with shape BCHW and we integrate over time.
        """
        solver = (solver or self.ode_solver).lower()
        steps = steps or self.sampling_timesteps

        assert z.shape == shape, f"Expected z of shape {shape}, got {z.shape}"
        x = z.to(self.device)
        B = x.shape[0]

        ##################################################################
        # TODO 4.1: Implement time grid and time step size,
        # similar to sample() function above.
        ##################################################################
        # Uniform time grid on [0, 1]
        ts = self._time_grid(steps)
        dt = float(1.0 / (steps - 1))
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

        for i in range(steps - 1):
            t_curr = ts[i].expand(B)
            if solver == "euler":
                x = self.ode_euler_step(x, t_curr, dt)
            else:
                raise ValueError(f"Unknown ODE solver: {solver}")
        
        img = unnormalize_to_zero_to_one(x)
        return img
