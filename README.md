# Physics-Informed Neural Network for the 1D Heat Equation

This repository implements a forward physics-informed neural network for the one-dimensional heat equation using PyTorch.

The inverse PINN for thermal-diffusivity identification will be added separately.

## Problem

The heat equation is

[
u_t-\alpha u_{xx}=0,
\qquad 0\leq x\leq 1,
\qquad 0\leq t\leq 1,
]

with thermal diffusivity

[
\alpha=0.1.
]

The initial condition is

[
u(x,0)=\sin(\pi x),
]

and the boundary conditions are

[
u(0,t)=0,
\qquad
u(1,t)=0.
]

The exact solution is

[
u(x,t)=e^{-\alpha\pi^2t}\sin(\pi x).
]

## PINN Formulation

The neural network approximates

[
u(x,t)\approx u_\theta(x,t).
]

The PDE residual is

[
f_\theta(x,t)
=============

## \frac{\partial u_\theta}{\partial t}

\alpha
\frac{\partial^2 u_\theta}{\partial x^2}.
]

The total training loss is

[
\mathcal{L}
===========

\mathcal{L}*{\mathrm{PDE}}
+
\mathcal{L}*{\mathrm{IC}}
+
\mathcal{L}_{\mathrm{BC}}.
]

The derivatives are computed using PyTorch automatic differentiation.

## Network and Training Parameters

* Input variables: (x) and (t)
* Output variable: (u(x,t))
* Hidden layers: 4
* Neurons per hidden layer: 32
* Activation function: `tanh`
* PDE collocation points: 10,000
* Initial-condition points: 100
* Boundary points: 100
* Optimizer: Adam
* Learning rate: (10^{-3})
* Training epochs: 5,000
* Random seed: 1234

## Repository Structure

```text
heat-equation-pinn/
├── forward_heat_pinn.py
├── requirements.txt
├── README.md
├── LICENSE
└── results/
    └── forward/
        ├── metrics.txt
        ├── loss_history.png
        └── solution_comparison.png
```

## Installation

```bash
git clone https://github.com/suvendu-nayak-research/heat-equation-pinn.git
cd heat-equation-pinn
pip install -r requirements.txt
```

## Run

```bash
python forward_heat_pinn.py
```

The program automatically creates the output directory

```text
results/forward/
```

and saves the error metrics and figures.

## Results

The final forward-PINN errors are:

| Metric                 |                   Value |
| ---------------------- | ----------------------: |
| Relative (L_2) error   | (2.010477\times10^{-3}) |
| Maximum absolute error | (8.217622\times10^{-3}) |
| Mean absolute error    | (6.399890\times10^{-4}) |
| RMSE                   | (9.385468\times10^{-4}) |

### Exact Solution, PINN Solution, and Absolute Error

![Solution comparison](results/forward/solution_comparison.png)

### Training History

![Training history](results/forward/loss_history.png)

## License

This project is released under the MIT License.
