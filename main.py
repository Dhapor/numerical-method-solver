import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="Numerical Methods Solver", layout="centered")

# Sidebar navigation
image = Image.open("IMG1.jpg")
st.sidebar.image(image, use_container_width=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Tutorial", "Solver"])
if page == "Overview":
    st.image('IMG2.jpg',  width = 600)
    st.title("\U0001F4D8 Numerical Methods Solver")
    st.markdown("""
    #### 👋 Welcome!
    This app helps you understand and solve **ordinary differential equations** (ODEs) using numerical methods.

    ---
    #### 🔍 What You Can Do:
    - Solve **ordinary differential equations (ODEs)** using methods like Euler, Runge-Kutta, Milne, and more
    - Solve **systems of linear equations** using Gauss Elimination, Gauss-Jordan, Gauss-Seidel, Jacobi, and LU Decomposition
    - Visualize **step-by-step calculations**
    - Input your own function, step size, and initial conditions
    - Get results up to 4–5 decimal places

    ---
    #### 📌 Who Is This For?
    - Students studying **Numerical Analysis** or **Linear Algebra**
    - Educators who want a visual teaching tool
    - Anyone learning how numerical methods work under the hood

    ---
    #### 💡 Supported Methods (So Far)
    **ODE Methods:**
    - Euler's Method (Explicit)
    - Euler's Method (Implicit)
    - Heun’s Method (Improved Euler)
    - Runge-Kutta 4th Order (RK4)
    - Taylor Series Method (1st order)
    - Milne's Predictor-Corrector Method

    **Linear Algebraic Methods:**
    - Gauss Elimination Method
    - Gauss-Jordan Method
    - Gauss-Seidel Iterative Method
    - Jacobi Iteration Method
    - LU Decomposition Method

    ---
    #### 🧠 How It Works:
    For ODEs:
    - Provide a function of `x` and `y`
    - Set initial values `x₀`, `y₀`, step size `h`, and number of steps `n`

    For Linear Equations:
    - Provide matrix `A` and vector `b` from your system of equations

    The app will:
    - Compute approximated values step by step
    - Display intermediate and final results
    """)

    st.markdown("### 🙏 Credits & Acknowledgements")
    st.markdown("""
    This project was developed with love and collaboration by:

    - **Datapsalm** - Core developer and UI designer  
    - **[Victoria, Fatima, Azeezat, Esther]** - Method contributions, debugging, and support  
    - **Inspiration:** This app was inspired by my tutorial instructor, **[SYSTEM]**, whose passion for teaching and dedication to our learning journey motivated me to build this platform   
    - Special thanks to mentors, testers, and everyone who gave feedback 
    - Built with **Streamlit**, powered by **Python**, and made for students everywhere 
    """)


elif page == "Solver":
    st.title("\U0001F9EE Numerical Methods Solver")

    algebraic_methods = [
        "Gauss Elimination Method",
        "Gauss-Jordan Method",
        "Gauss-Seidel Iterative Method",
        "Jacobi Iteration Method",
        "LU Decomposition Method"
    ]
    ode_methods = [
        "Euler (Explicit)",
        "Euler (Implicit)",
        "Heun’s Method (Improved Euler)",
        "Runge-Kutta 4th Order (RK4)",
        "Taylor Series Method",
        "Milne's Predictor-Corrector Method"
    ]

    method_type = st.radio("Choose a Method Category:", ["ODE Methods", "Algebraic Methods"])

    if method_type == "ODE Methods":
        method = st.selectbox("Choose a Numerical Method:", ode_methods)
        col1, col2 = st.columns(2)

        with col1:
            x0 = st.number_input("Initial x (x₀):", value=0.0, format="%.5f")
            h = st.number_input("Step size (h):", value=0.1, format="%.5f")

        with col2:
            f_str = st.text_input("Enter f(x, y):", "x + y")
            y0 = st.number_input("Initial y (y₀):", value=1.0, format="%.5f")
            n = st.number_input("Number of steps:", value=5, step=1)

        st.markdown("### 🔄 Click below to solve")
        compute = st.button("🔍 Compute Solution")

        if compute:
            try:
                f = lambda x, y: eval(f_str, {"x": x, "y": y, "math": math})
                x_vals = [x0]
                y_vals = [y0]
                x = x0
                y = y0
                results = [{"Step": 0, "x": round(x, 5), "y": round(y, 5)}]

                if method == "Milne's Predictor-Corrector Method":
                    # Generate first 3 points using RK4
                    k1 = h * f(x, y)
                    k2 = h * f(x + h/2, y + k1/2)
                    k3 = h * f(x + h/2, y + k2/2)
                    k4 = h * f(x + h, y + k3)
                    y1 = y + (k1 + 2*k2 + 2*k3 + k4) / 6
                    x1 = x + h

                    k1 = h * f(x1, y1)
                    k2 = h * f(x1 + h/2, y1 + k1/2)
                    k3 = h * f(x1 + h/2, y1 + k2/2)
                    k4 = h * f(x1 + h, y1 + k3)
                    y2 = y1 + (k1 + 2*k2 + 2*k3 + k4) / 6
                    x2 = x1 + h

                    k1 = h * f(x2, y2)
                    k2 = h * f(x2 + h/2, y2 + k1/2)
                    k3 = h * f(x2 + h/2, y2 + k2/2)
                    k4 = h * f(x2 + h, y2 + k3)
                    y3 = y2 + (k1 + 2*k2 + 2*k3 + k4) / 6
                    x3 = x2 + h

                    xs = [x, x1, x2, x3]
                    ys = [y, y1, y2, y3]
                    results = [{"Step": i, "x": round(xs[i], 5), "y": round(ys[i], 5)} for i in range(4)]

                    for i in range(4, int(n) + 1):
                        f_n3 = f(xs[i-3], ys[i-3])
                        f_n2 = f(xs[i-2], ys[i-2])
                        f_n1 = f(xs[i-1], ys[i-1])
                        f_n = f(xs[i-0], ys[i-0])

                        # Predictor
                        y_pred = ys[i-4] + (4*h/3)*(2*f_n2 - f_n1 + 2*f_n)
                        x_new = xs[i-1] + h
                        # Corrector
                        f_pred = f(x_new, y_pred)
                        y_corr = ys[i-2] + (h/3)*(f_n1 + 4*f_n + f_pred)

                        xs.append(x_new)
                        ys.append(y_corr)
                        results.append({"Step": i, "x": round(x_new, 5), "y": round(y_corr, 5)})
                else:
                    for i in range(1, int(n)+1):
                        if method == "Euler (Explicit)":
                            y = y + h * f(x, y)
                            x = x + h
                        elif method == "Euler (Implicit)":
                            y_new = y + h * f(x + h, y)
                            y = y_new
                            x = x + h
                        elif method == "Heun’s Method (Improved Euler)":
                            y_predict = y + h * f(x, y)
                            slope_avg = (f(x, y) + f(x + h, y_predict)) / 2
                            y = y + h * slope_avg
                            x = x + h
                        elif method == "Runge-Kutta 4th Order (RK4)":
                            k1 = h * f(x, y)
                            k2 = h * f(x + h/2, y + k1/2)
                            k3 = h * f(x + h/2, y + k2/2)
                            k4 = h * f(x + h, y + k3)
                            y += (k1 + 2*k2 + 2*k3 + k4) / 6
                            x += h
                        elif method == "Taylor Series Method":
                            y = y + h * f(x, y)
                            x = x + h

                        results.append({"Step": i, "x": round(x, 5), "y": round(y, 5)})
                        x_vals.append(x)
                        y_vals.append(y)

                st.subheader(f"📊 Results using {method}")
                df = pd.DataFrame(results)
                st.dataframe(df)
            except Exception as e:
                st.error(f"⚠️ Error in function input: {e}")

    elif method_type == "Algebraic Methods":
        method = st.selectbox("Choose a Linear Algebra Method:", algebraic_methods)

        st.write("Enter your coefficient matrix A and RHS vector b (1 equation per line).")
        A_str = st.text_area("Enter matrix A (e.g., 2 1 -1\\n-3 -1 2\\n-2 1 2):", "2 1 -1\n-3 -1 2\n-2 1 2")
        b_str = st.text_area("Enter vector b (e.g., 8\\n-11\\n-3):", "8\n-11\n-3")

        if st.button("🔍 Solve System"):
            try:
                A = np.array([list(map(float, row.strip().split())) for row in A_str.strip().split('\n')])
                b = np.array([float(num) for num in b_str.strip().split('\n')])
                n = len(b)

                if method == "Gauss Elimination Method":
                    for i in range(n):
                        for j in range(i + 1, n):
                            factor = A[j][i] / A[i][i]
                            A[j] = A[j] - factor * A[i]
                            b[j] = b[j] - factor * b[i]
                    x_sol = np.zeros(n)
                    for i in range(n - 1, -1, -1):
                        x_sol[i] = (b[i] - np.dot(A[i][i + 1:], x_sol[i + 1:])) / A[i][i]
                elif method == "Gauss-Jordan Method":
                    aug = np.hstack((A, b.reshape(-1,1)))
                    for i in range(n):
                        aug[i] = aug[i] / aug[i][i]
                        for j in range(n):
                            if i != j:
                                aug[j] = aug[j] - aug[j][i] * aug[i]
                    x_sol = aug[:, -1]
                elif method == "Gauss-Seidel Iterative Method":
                    x_sol = np.zeros(n)
                    for _ in range(25):
                        for i in range(n):
                            x_sol[i] = (b[i] - np.dot(A[i, :i], x_sol[:i]) - np.dot(A[i, i+1:], x_sol[i+1:])) / A[i, i]
                elif method == "Jacobi Iteration Method":
                    x_sol = np.zeros(n)
                    for _ in range(25):
                        x_new = np.copy(x_sol)
                        for i in range(n):
                            x_new[i] = (b[i] - np.dot(A[i, :i], x_sol[:i]) - np.dot(A[i, i+1:], x_sol[i+1:])) / A[i, i]
                        x_sol = x_new
                elif method == "LU Decomposition Method":
                    L = np.zeros_like(A)
                    U = np.zeros_like(A)
                    for i in range(n):
                        L[i][i] = 1
                        for j in range(i, n):
                            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
                        for j in range(i + 1, n):
                            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
                    y = np.zeros(n)
                    for i in range(n):
                        y[i] = b[i] - np.dot(L[i, :i], y[:i])
                    x_sol = np.zeros(n)
                    for i in range(n - 1, -1, -1):
                        x_sol[i] = (y[i] - np.dot(U[i, i+1:], x_sol[i+1:])) / U[i][i]

                df_result = pd.DataFrame({"Variable": [f"x{i + 1}" for i in range(n)], "Value": x_sol})
                st.success("✅ System Solved Successfully")
                st.dataframe(df_result)

            except Exception as e:
                st.error(f"⚠️ Error in matrix input: {e}")

elif page == "Tutorial":
    st.title("📘 How to Use This Solver")
    st.markdown("---")

    # ODE Section
    st.markdown("## 🔢 ODE Methods Input")
    st.markdown("These methods solve ordinary differential equations like:")
    st.code("dy/dx = f(x, y)", language="python")

    st.markdown("### ✍️ What You Need to Enter")
    st.markdown("""
    - **f(x, y)** — Your function to solve. Enter it using Python-style math:
        - Example: `x + y`, `x * y`, `x / y`, `(x + y) ** 2`
        - Always use `**` for powers (instead of `^`)
        - Wrap expressions with parentheses when needed
        - To use square roots or trigonometric functions, prefix with `math.` like `math.sqrt(x)`, `math.sin(x)`
    """)

    st.markdown("---")
    st.markdown("### ✍️ Python Math Syntax Reference (for Function Input)")

    st.markdown("""
    | Math Operation       | Python Syntax         | Example                    |
    |----------------------|------------------------|----------------------------|
    | Addition             | `+`                    | `x + y`                    |
    | Subtraction          | `-`                    | `x - y`                    |
    | Multiplication       | `*`                    | `x * y`                    |
    | Division             | `/`                    | `x / y`                    |
    | Exponent (Power)     | `**`                   | `x ** 2` (means x squared) |
    | Square Root          | `math.sqrt(x)`         | `math.sqrt(x + y)`         |
    | Exponential (e^x)    | `math.exp(x)`          | `math.exp(x)`              |
    | Natural Log (ln x)   | `math.log(x)`          | `math.log(x + 1)`          |
    | Sine                 | `math.sin(x)`          | `math.sin(x)`              |
    | Cosine               | `math.cos(x)`          | `math.cos(x)`              |
    | Tangent              | `math.tan(x)`          | `math.tan(x)`              |
    | Absolute Value       | `math.fabs(x)`         | `math.fabs(x - y)`         |

    ✅ Always include `math.` before functions like sin, sqrt, log, etc.
    """)

    st.markdown("### 🧪 Other Inputs")
    st.markdown("""
    - `Initial x₀` — e.g. `0`
    - `Initial y₀` — e.g. `1`
    - `Step size (h)` — e.g. `0.1`
    - `Number of steps (n)` — e.g. `5`
    """)

    st.markdown("---")

    # Algebraic Section
    st.markdown("## 🧮 Algebraic Methods Input")
    st.markdown("These methods solve systems like:")
    st.code("""
    2x1 + 1x2 - 1x3 = 8
    -3x1 - 1x2 + 2x3 = -11
    -2x1 + 1x2 + 2x3 = -3
    """, language="text")

    st.markdown("### ✍️ What You Need to Enter")
    st.markdown("""
    - **Matrix A** — Coefficients of your system:
      ```
      2 1 -1
      -3 -1 2
      -2 1 2
      ```

    - **Vector b** — Constants on the right-hand side:
      ```
      8
      -11
      -3
      ```

    ✅ Make sure:
    - Matrix A has the same number of rows and columns (i.e., square)
    - Vector b has the same number of rows as Matrix A
    - No blank lines or extra spaces
    """)

    st.markdown("---")
    st.info("📏 Tip: Use clean inputs and check that dimensions match for accurate results.")
    st.success("You're ready! Head to the **Solver** tab to try it out.")
    st.markdown("---")

