import { useState } from "react";
import { AmbientBackground } from "./components/AmbientBackground";
import { BurnoutForm } from "./components/BurnoutForm";
import { ResultPanel } from "./components/ResultPanel";
import { getBurnoutPrediction } from "./api/predict";
import type { BurnoutRequest, BurnoutResponse } from "./types/burnout";
import "./App.css";

export default function App() {
  const [result, setResult] = useState<BurnoutResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(data: BurnoutRequest) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await getBurnoutPrediction(data);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error occurred.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <AmbientBackground />
      <header className="app-header">
        <h1 className="app-title">BurnOut Lens</h1>
        <p className="app-subtitle">
          Predict employee burnout risk and understand what's driving it —
          powered by XGBoost + SHAP explanations.
        </p>
      </header>

      <main className="app-main">
        <section className="app-form-section">
          <BurnoutForm onResult={handleSubmit} loading={loading} />
        </section>

        <section className="app-result-section">
          {loading && (
            <div className="app-loading">
              <div className="spinner" />
              <span>Running prediction…</span>
            </div>
          )}
          {error && (
            <div className="app-error">
              <strong>Error:</strong> {error}
            </div>
          )}
          {result && !loading && <ResultPanel result={result} />}
          {!result && !loading && !error && (
            <div className="app-placeholder">
              <p>
                Adjust the sliders and click <strong>Predict burnout risk</strong> to see your result.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
