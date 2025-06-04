import axios from "axios";
import { useEffect, useState } from 'react';
import Plot from "react-plotly.js";

function App() {
  const [data, setData] = useState(null);
  const [func, setFunc] = useState("sin");

  useEffect(() => {
    axios.get(`http://localhost:8000/fft?f=${func}&start=0&stop=10&points=100`).then((res) => {
      setData(res.data);
    });
  }, [func]);


  if (!data) return <div>Lade...</div>
  return (
    <div>
      <div style={{ textAlign: "center", marginBottom: "1rem" }}>
        <h1>Fourier Visualization</h1>
        <select value={func} onChange={(e) => setFunc(e.target.value)}>
          <option value="sin">sin(x)</option>
          <option value="cos">cos(x)</option>
          <option value="square">square wave</option>
        </select>
      </div>
      <Plot
        data={[
          { x: data.x, y: data.f, type: "scatter", name: "f(x)" },
        ]}
        layout={{ title: `f(x) = ${func}(x)` }}
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
