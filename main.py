import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/fft")
def compute_fft():
    x = jnp.linspace(0, 10, 100)
    y = jnp.sin(x)
    y_fourier = jnp.fft.rfft(y)
    freqs = jnp.fft.rfftfreq(len(x), d=(x[1] - x[0]))
    return {
        "x": x.tolist(),
        "f": y.tolist(),
        "freqs": freqs.tolist(),
        "fft_abs": jnp.abs(y_fourier).tolist(),
    }
