# 📘 Numerical Methods Solver

An interactive Streamlit app to solve **Ordinary Differential Equations (ODEs)** and **Systems of Linear Equations** using popular numerical methods.

---

## 🔗 Live App

🚀 Hosted on Streamlit Cloud:  
[https://numerical-methods.streamlit.app](https://numerical-methods.streamlit.app)  


---

## ✨ Features

- 📐 **ODE Solvers**
  - Euler's Method (Explicit & Implicit)
  - Heun’s Method (Improved Euler)
  - Runge-Kutta 4th Order (RK4)
  - Taylor Series (1st Order)
  - Milne's Predictor-Corrector

- 🔢 **Linear Algebra Solvers**
  - Gauss Elimination
  - Gauss-Jordan
  - Gauss-Seidel Iteration
  - Jacobi Iteration
  - LU Decomposition

- ✍️ Custom inputs: equations, step size, initial conditions, matrix data
- 📊 Clean layout and tabular outputs for easy interpretation

---

## 📥 Installation

Run locally using:

```bash
git clone https://github.com/Dhapor/numerical-methods-solver.git
cd numerical-methods-solver
pip install -r requirements.txt
streamlit run app.py
