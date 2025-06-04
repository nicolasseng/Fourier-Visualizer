from typing import Literal

import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sympy import Symbol, lambdify, sympify

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/fft")
def compute_fft(
    f: Literal["sin", "cos", "square", "custom"],
    start: float,
    stop: float,
    points: int,
    custom: str = "",
):
    if start >= stop:
        stop = start + 1
    x = jnp.linspace(start, stop, points)
    if f == "sin":
        y = jnp.sin(x)
    elif f == "cos":
        y = jnp.cos(x)
    elif f == "square":
        y = jnp.sign(jnp.sin(x))
    elif f == "custom":
        try:
            x_sym = Symbol("x")
            expr = sympify(custom)
            f_np = lambdify(x_sym, expr, "numpy")
            f_vals = f_np(np.array(x))
            f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
            y = jnp.array(f_vals)
        except Exception as e:
            return {"Error": f"Invalid function: {e}"}
    else:
        return {"Error": "Unknown Function"}

    y_fourier = jnp.fft.rfft(y)
    freqs = jnp.fft.rfftfreq(len(x), d=(x[1] - x[0]))
    return {
        "x": x.tolist(),
        "f": y.tolist(),
        "freqs": freqs.tolist(),
        "fft_abs": jnp.abs(y_fourier).tolist(),
    }
