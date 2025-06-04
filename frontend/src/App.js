import axios from "axios";
import { useEffect, useState } from 'react';
import Plot from "react-plotly.js";

function App() {
  const [data, setData] = useState(null);
  const [func, setFunc] = useState("sin");
  const [start, setStart] = useState(0);
  const [stop, setStop] = useState(10);
  const [num_points, setPoints] = useState(100);
  const [custom_Func, setCustom] = useState(null);

  useEffect(() => {
    axios.get(`http://localhost:8000/fft?f=${func}&start=${parseFloat(start)}&stop=${parseFloat(stop)}&points=${parseInt(num_points)}&custom=${encodeURIComponent(custom_Func)}`)
      .then((res) => setData(res.data))
      .catch(err => console.error(err));
  }, [func, start, stop, num_points, custom_Func]);


  if (!data) return <div>Loading...</div>
  return (
    <div>
      <div style={{ textAlign: "center", marginBottom: "1rem" }}>

        <h1>Fourier Visualization</h1>

        <label>
          Function:
          <select value={func} onChange={(e) => setFunc(e.target.value)}>
            <option value="sin">sin(x)</option>
            <option value="cos">cos(x)</option>
            <option value="square">square wave</option>
            <option value="custom">custom function</option>
          </select>
        </label>

        <label>
          Custom Function:
          <input
            type="text"
            value={custom_Func}
            onChange={e => setCustom(e.target.value)}
          />
        </label>

        <label>
          Start:
          <input
            type="number"
            value={start}
            onChange={e => setStart(e.target.value)}
            step="any"
          />
        </label>


        <label>
          Stop:
          <input
            type="number"
            value={stop}
            onChange={e => setStop(e.target.value)}
            step="any"
          />
        </label>


        <label>
          Number of points:
          <input
            type="number"
            value={num_points}
            onChange={e => setPoints(e.target.value)}
            step="1"
            min="1"
          />
        </label>

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
