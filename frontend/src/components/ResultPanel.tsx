import type { BurnoutResponse, BurnoutLevel } from "../types/burnout";
import { FEATURE_LABELS } from "../types/burnout";
import "./ResultPanel.css";

interface Props {
  result: BurnoutResponse;
}

// Color palette per severity level — simple, no theming system needed
const LEVEL_COLORS: Record<BurnoutLevel, { bg: string; text: string; border: string }> = {
  Low:      { bg: "#e8f5e9", text: "#2e7d32", border: "#4caf50" },
  Moderate: { bg: "#fff8e1", text: "#f57f17", border: "#ffb300" },
  High:     { bg: "#fff3e0", text: "#e65100", border: "#ff6d00" },
  Severe:   { bg: "#ffebee", text: "#b71c1c", border: "#f44336" },
};

// Max absolute SHAP value among contributions — used to normalise bar widths
function maxAbsShap(contributions: BurnoutResponse["contributions"]): number {
  return Math.max(...contributions.map((c) => Math.abs(c.shap_value)), 0.001);
}

export function ResultPanel({ result }: Props) {
  const { burnout_score, burnout_level, base_value, contributions } = result;
  const colors = LEVEL_COLORS[burnout_level];
  const maxShap = maxAbsShap(contributions);

  return (
    <div className="result-panel">
      {/* Score header */}
      <div
        className="result-panel__score-card"
        style={{
          backgroundColor: colors.bg,
          borderColor: colors.border,
        }}
      >
        <div className="result-panel__score-number" style={{ color: colors.text }}>
          {burnout_score.toFixed(1)}<span className="result-panel__score-denom"> / 10</span>
        </div>
        <div
          className="result-panel__level-badge"
          style={{ backgroundColor: colors.border, color: "#fff" }}
        >
          {burnout_level}
        </div>
        <div className="result-panel__base-value">
          Model baseline: {base_value.toFixed(2)}
        </div>
      </div>

      {/* Explanation section */}
      <div className="result-panel__explanation">
        <h3 className="result-panel__section-title">Why this score?</h3>
        <p className="result-panel__section-subtitle">
          Factors sorted by influence (largest first). Bars show relative impact.
        </p>

        <ul className="contributions-list">
          {contributions.map((c) => {
            const isIncrease = c.direction === "increases";
            const barPct = (Math.abs(c.shap_value) / maxShap) * 100;
            const featureLabel =
              (FEATURE_LABELS as Record<string, string>)[c.feature] ?? c.feature;

            return (
              <li key={c.feature} className="contribution-item">
                <div className="contribution-item__header">
                  <span className="contribution-item__name">{featureLabel}</span>
                  <span className="contribution-item__value">
                    {c.value % 1 === 0 ? c.value : c.value.toFixed(1)}
                  </span>
                  <span
                    className={`contribution-item__direction contribution-item__direction--${c.direction}`}
                  >
                    {isIncrease ? "▲ increases" : "▼ decreases"}
                  </span>
                  <span className="contribution-item__shap">
                    {isIncrease ? "+" : ""}{c.shap_value.toFixed(3)}
                  </span>
                </div>
                <div className="contribution-item__bar-track">
                  <div
                    className={`contribution-item__bar contribution-item__bar--${c.direction}`}
                    style={{ width: `${barPct}%` }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
