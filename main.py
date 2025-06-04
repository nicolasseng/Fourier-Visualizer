import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/fft")
def compute_fft(f: str, start: float, stop: float, points: int):
    x = jnp.linspace(start, stop, points)
    if f == "sin":
        y = jnp.sin(x)
    elif f == "cos":
        y = jnp.cos(x)
    elif f == "square":
        y = jnp.sign(jnp.sin(x))
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
