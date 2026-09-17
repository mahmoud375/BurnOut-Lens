import type { BurnoutResponse, BurnoutLevel } from "../types/burnout";
import { FEATURE_LABELS } from "../types/burnout";
import "./ResultPanel.css";

interface Props {
  result: BurnoutResponse;
}

// Subtle border/tone mapping per severity level matching Integrated Biosciences
const LEVEL_BADGE_CLASS: Record<BurnoutLevel, string> = {
  Low: "level-badge--low",
  Moderate: "level-badge--moderate",
  High: "level-badge--high",
  Severe: "level-badge--severe",
};

// Max absolute SHAP value among contributions — used to normalise bar widths
function maxAbsShap(contributions: BurnoutResponse["contributions"]): number {
  return Math.max(...contributions.map((c) => Math.abs(c.shap_value)), 0.001);
}

export function ResultPanel({ result }: Props) {
  const { burnout_score, burnout_level, base_value, contributions } = result;
  const maxShap = maxAbsShap(contributions);

  return (
    <div className="result-panel">
      {/* Score card: Clean, border-delimited container (1px solid #c9cbbe, radius 16px) */}
      <div className="result-panel__score-card">
        <div className="result-panel__score-number">
          {burnout_score.toFixed(1)}
          <span className="result-panel__score-denom"> / 10</span>
        </div>
        <div>
          <span className={`result-panel__level-badge ${LEVEL_BADGE_CLASS[burnout_level]}`}>
            {burnout_level} risk
          </span>
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
