# Fourier Visualizer

## Overview

The Fourier Visualizer is a web application that lets you see various functions and their corresponding Fourier spectra. It's made up of a **React-based frontend** for user interaction and a **FastAPI-based Python backend** for performing the Fourier Transform (FFT) calculations. The entire application is packaged with Docker and Docker Compose, making it a breeze to set up and run.

---

## Features

* **Function Selection:** Pick from pre-defined functions like sine, cosine, and square wave.
* **Custom Function Input:** Define your own math functions to visualize.
* **Parameter Control:** Easily adjust the start, stop, and number of points for your function plot.
* **Real-time Visualization:** Watch your function plot and its Fourier spectrum update instantly.
* **Dockerized Environment:** Run the whole application with simple Docker Compose commands.

---

## Technologies Used

* **Frontend:** React, Axios, Plotly.js
* **Backend:** FastAPI, Uvicorn, JaxNumPy (for FFT calculations)
* **Containerization:** Docker, Docker Compose

---

## Getting Started

Follow these steps to get the project up and running on your local machine for development and testing.

### Prerequisites

You'll need Docker and Docker Compose installed on your system:

* [**Docker Desktop**](https://www.docker.com/products/docker-desktop/) (recommended for Windows and macOS)
* [**Docker Engine & Docker Compose**](https://docs.docker.com/engine/install/) (for Linux)

### Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/nicolasseng/Fourier-Visualizer.git
    cd Fourier-Visualizer
    ```

2.  **Navigate to the project root:**

    Make sure you're in the directory that contains `docker-compose.yml` and the `frontend` and `backend` folders.

3.  **Build and run the Docker containers:**

    This command will build the Docker images for both your frontend and backend, then start the services. The `--build` flag is vital here; it ensures any changes in your Dockerfiles or environment variables are applied.

    ```bash
    docker-compose up --build
    ```

    You'll see logs from both services. The frontend might show some warnings during compilation, but it should still start up.

4.  **Access the application:**

    Once the services are running, open your web browser and go to:

    ```
    http://localhost:3000
    ```

    You should now see the Fourier Visualizer application.

---
