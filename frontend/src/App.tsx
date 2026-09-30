import { useState, useEffect } from "react";
import { AmbientBackground } from "./components/AmbientBackground";
import { BurnoutForm } from "./components/BurnoutForm";
import { ResultPanel } from "./components/ResultPanel";
import { getBurnoutPrediction } from "./api/predict";
import type { BurnoutRequest, BurnoutResponse } from "./types/burnout";
import "./App.css";

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  try {
    const saved = localStorage.getItem("burnout_lens_theme");
    if (saved === "light" || saved === "dark") {
      return saved;
    }
  } catch {
    // Storage access might be restricted
  }

  if (
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  ) {
    return "dark";
  }

  return "light";
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [result, setResult] = useState<BurnoutResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("burnout_lens_theme", theme);
    } catch {
      // Ignore storage errors
    }
  }, [theme]);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (e: MediaQueryListEvent) => {
      try {
        const saved = localStorage.getItem("burnout_lens_theme");
        if (!saved) {
          setTheme(e.matches ? "dark" : "light");
        }
      } catch {
        // Ignore storage errors
      }
    };
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

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
        <button
          type="button"
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {theme === "dark" ? (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <circle cx="12" cy="12" r="5" />
              <line x1="12" y1="1" x2="12" y2="3" />
              <line x1="12" y1="21" x2="12" y2="23" />
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
              <line x1="1" y1="12" x2="3" y2="12" />
              <line x1="21" y1="12" x2="23" y2="12" />
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </button>
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
