from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 1. SETTINGS
# ============================================================
SEED = 1234

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = Path("results/forward")
output_dir.mkdir(parents=True, exist_ok=True)

print("Using device:", device)


# ============================================================
# 2. PROBLEM AND TRAINING PARAMETERS
# ============================================================
alpha = 0.1

x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

N_f = 10000
N_ic = 100
N_bc = 100

epochs = 5000
learning_rate = 0.001


# ============================================================
# 3. NEURAL NETWORK
# ============================================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x, t):
        inputs = torch.cat((x, t), dim=1)

        hidden = torch.tanh(self.layer1(inputs))
        hidden = torch.tanh(self.layer2(hidden))
        hidden = torch.tanh(self.layer3(hidden))
        hidden = torch.tanh(self.layer4(hidden))

        return self.output_layer(hidden)


# ============================================================
# 4. PDE RESIDUAL
# ============================================================
def compute_pde_residual(model, x, t):
    u = model(x, t)

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    return u_t - alpha * u_xx


# ============================================================
# 5. INITIAL AND BOUNDARY POINTS
# ============================================================
x_ic = torch.linspace(
    x_min,
    x_max,
    N_ic,
    device=device
).reshape(-1, 1)

t_ic = torch.zeros_like(x_ic)
u_ic_exact = torch.sin(torch.pi * x_ic)

t_bc = torch.linspace(
    t_min,
    t_max,
    N_bc,
    device=device
).reshape(-1, 1)

x_bc_left = torch.full_like(t_bc, x_min)
x_bc_right = torch.full_like(t_bc, x_max)


# ============================================================
# 6. MODEL AND OPTIMIZER
# ============================================================
model = PINN().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

total_loss_history = []
pde_loss_history = []
initial_loss_history = []
boundary_loss_history = []


# ============================================================
# 7. TRAINING
# ============================================================
print("\nTraining started...")

model.train()

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    x_f = (
        x_min
        + (x_max - x_min) * torch.rand(N_f, 1, device=device)
    ).requires_grad_(True)

    t_f = (
        t_min
        + (t_max - t_min) * torch.rand(N_f, 1, device=device)
    ).requires_grad_(True)

    residual = compute_pde_residual(model, x_f, t_f)
    loss_pde = torch.mean(residual**2)

    u_ic_predicted = model(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_predicted - u_ic_exact)**2)

    u_bc_left_predicted = model(x_bc_left, t_bc)
    u_bc_right_predicted = model(x_bc_right, t_bc)

    loss_bc = (
        torch.mean(u_bc_left_predicted**2)
        + torch.mean(u_bc_right_predicted**2)
    )

    total_loss = loss_pde + loss_ic + loss_bc

    total_loss.backward()
    optimizer.step()

    total_loss_history.append(total_loss.item())
    pde_loss_history.append(loss_pde.item())
    initial_loss_history.append(loss_ic.item())
    boundary_loss_history.append(loss_bc.item())

    if epoch == 1 or epoch % 500 == 0:
        print(
            f"Epoch {epoch:5d}/{epochs} | "
            f"Total: {total_loss.item():.6e} | "
            f"PDE: {loss_pde.item():.6e} | "
            f"IC: {loss_ic.item():.6e} | "
            f"BC: {loss_bc.item():.6e}"
        )

print("Training finished.")


# ============================================================
# 8. EVALUATION
# ============================================================
Nx = 256
Nt = 101

x_test = np.linspace(x_min, x_max, Nx)
t_test = np.linspace(t_min, t_max, Nt)

X_grid, T_grid = np.meshgrid(x_test, t_test)

x_test_tensor = torch.tensor(
    X_grid.reshape(-1, 1),
    dtype=torch.float32,
    device=device
)

t_test_tensor = torch.tensor(
    T_grid.reshape(-1, 1),
    dtype=torch.float32,
    device=device
)

model.eval()

with torch.no_grad():
    u_predicted = model(
        x_test_tensor,
        t_test_tensor
    ).cpu().numpy().reshape(Nt, Nx)

u_exact = (
    np.exp(-alpha * np.pi**2 * T_grid)
    * np.sin(np.pi * X_grid)
)

absolute_error = np.abs(u_predicted - u_exact)


# ============================================================
# 9. ERROR METRICS
# ============================================================
error_vector = (u_predicted - u_exact).reshape(-1)
exact_vector = u_exact.reshape(-1)

relative_l2_error = (
    np.linalg.norm(error_vector)
    / np.linalg.norm(exact_vector)
)

maximum_absolute_error = np.max(np.abs(error_vector))
mean_absolute_error = np.mean(np.abs(error_vector))
rmse = np.sqrt(np.mean(error_vector**2))

metrics_text = (
    "Forward PINN: One-Dimensional Heat Equation\n"
    "===========================================\n"
    f"Thermal diffusivity alpha : {alpha:.6f}\n"
    f"Relative L2 error         : {relative_l2_error:.6e}\n"
    f"Maximum absolute error    : {maximum_absolute_error:.6e}\n"
    f"Mean absolute error       : {mean_absolute_error:.6e}\n"
    f"RMSE                      : {rmse:.6e}\n"
)

print("\n" + metrics_text)

(output_dir / "metrics.txt").write_text(
    metrics_text,
    encoding="utf-8"
)


# ============================================================
# 10. PLOTS AND MODEL
# ============================================================
plt.figure(figsize=(8, 5))

plt.semilogy(total_loss_history, label="Total loss")
plt.semilogy(pde_loss_history, label="PDE loss")
plt.semilogy(initial_loss_history, label="Initial-condition loss")
plt.semilogy(boundary_loss_history, label="Boundary-condition loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PINN Training History")
plt.legend()
plt.tight_layout()

plt.savefig(
    output_dir / "loss_history.png",
    dpi=300
)

plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_data = [
    u_exact,
    u_predicted,
    absolute_error
]

plot_titles = [
    "Exact Solution",
    "PINN Solution",
    "Absolute Error"
]

for axis, data, title in zip(axes, plot_data, plot_titles):
    image = axis.pcolormesh(
        X_grid,
        T_grid,
        data,
        shading="auto"
    )

    axis.set_xlabel("x")
    axis.set_ylabel("t")
    axis.set_title(title)

    fig.colorbar(image, ax=axis)

plt.tight_layout()

plt.savefig(
    output_dir / "solution_comparison.png",
    dpi=300
)

plt.close()

torch.save(
    model.state_dict(),
    output_dir / "heat_equation_pinn.pt"
)

print("Saved files:")
print("1. results/forward/metrics.txt")
print("2. results/forward/loss_history.png")
print("3. results/forward/solution_comparison.png")
print("4. results/forward/heat_equation_pinn.pt")
