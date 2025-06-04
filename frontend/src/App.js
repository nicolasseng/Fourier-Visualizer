import axios from "axios";
import { useEffect, useState } from 'react';
import Plot from "react-plotly.js";

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get("http://localhost:8000/fft").then((res) => {
      setData(res.data);
    });
  }, []);

  if (!data) return <div>Lade...</div>
  return (
    <div>
      <h1>Fourier Demo</h1>
      <Plot
        data={[
          { x: data.x, y: data.f, type: "scatter", name: "f(x)" },
        ]}
        layout={{ title: "f(x) = sin(x)" }}
      />
      <Plot
        data={[
          { x: data.freqs, y: data.fft_abs, type: "bar", name: "|F(k)|" },
        ]}
        layout={{ title: "Fourier-Spektrum" }}
      />
    </div>
  );
}

export default App;
