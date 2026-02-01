"use client";
import { useState } from "react";

export default function Home() {
  const [age, setAge] = useState(10);
  const [sex, setSex] = useState(0);
  const [jsonInput, setJsonInput] = useState('{\n  "Basic_Demos-Enroll_Season": "Spring",\n  "CGAS-Season": "Summer",\n  "Physical-BMI": 18.5\n}');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      let additionalFeatures = {};
      try {
        additionalFeatures = JSON.parse(jsonInput);
      } catch (err) {
        throw new Error("Invalid JSON in Additional Features");
      }

      const payload = {
        features: {
          "Basic_Demos-Age": Number(age),
          "Basic_Demos-Sex": Number(sex),
          ...additionalFeatures,
        },
      };

      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Prediction failed");
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-lg shadow-md">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Model Prediction
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Enter features to get a prediction.
          </p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700">
                Age
              </label>
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(Number(e.target.value))}
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                required
              />
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700">
                Sex (0=F, 1=M)
              </label>
              <select
                value={sex}
                onChange={(e) => setSex(Number(e.target.value))}
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
              >
                <option value={0}>Female</option>
                <option value={1}>Male</option>
              </select>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700">
                Additional Features (JSON)
              </label>
              <textarea
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                rows={10}
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm font-mono"
              />
            </div>
          </div>

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400"
            >
              {loading ? "Processing..." : "Predict"}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-md">
            Error: {error}
          </div>
        )}

        {result && (
          <div className="mt-6 p-4 bg-green-50 rounded-md border border-green-200">
            <h3 className="text-lg font-medium text-green-900">Result</h3>
            <p className="mt-2 text-sm text-green-700">
              Raw Prediction: <strong>{result.prediction_raw.toFixed(4)}</strong>
            </p>
            <p className="mt-1 text-sm text-green-700">
              Class: <strong>{result.prediction_class}</strong>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
