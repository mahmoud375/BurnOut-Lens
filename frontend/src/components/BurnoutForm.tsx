import { useState, type FormEvent } from "react";
import type { BurnoutRequest, SeniorityLevel } from "../types/burnout";
import {
  FEATURE_BOUNDS,
  FEATURE_LABELS,
  SENIORITY_LEVELS,
} from "../types/burnout";
import "./BurnoutForm.css";

interface Props {
  onResult: (data: BurnoutRequest) => void;
  loading: boolean;
}

// Sensible defaults — a mid-career employee with average stress indicators
const DEFAULTS: BurnoutRequest = {
  seniority_level:          "Senior",
  work_hours_per_week:      45,
  meetings_per_day:         4,
  sleep_hours_per_night:    7,
  exercise_days_per_week:   3,
  vacation_days_taken:      10,
  social_support_score:     6,
  manager_support_score:    6,
  deadline_pressure_score:  6,
};

export function BurnoutForm({ onResult, loading }: Props) {
  const [values, setValues] = useState<BurnoutRequest>(DEFAULTS);

  function handleChange(
    field: keyof BurnoutRequest,
    value: string | number
  ) {
    setValues((prev) => ({ ...prev, [field]: value }));
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    onResult(values);
  }

  // Numeric fields in display order (seniority handled separately as select)
  const numericFields = (
    Object.keys(FEATURE_BOUNDS) as Array<
      Exclude<keyof BurnoutRequest, "seniority_level">
    >
  );

  return (
    <form className="burnout-form" onSubmit={handleSubmit} noValidate>
      {/* Seniority — select */}
      <div className="form-field">
        <label htmlFor="seniority_level">
          {FEATURE_LABELS.seniority_level}
        </label>
        <select
          id="seniority_level"
          value={values.seniority_level}
          onChange={(e) =>
            handleChange("seniority_level", e.target.value as SeniorityLevel)
          }
          disabled={loading}
        >
          {SENIORITY_LEVELS.map((lvl) => (
            <option key={lvl} value={lvl}>
              {lvl}
            </option>
          ))}
        </select>
      </div>

      {/* Numeric fields */}
      {numericFields.map((field) => {
        const { min, max, step } = FEATURE_BOUNDS[field];
        const val = values[field] as number;
        return (
          <div className="form-field" key={field}>
            <label htmlFor={field}>
              {FEATURE_LABELS[field]}
              <span className="form-field__value">{val}</span>
            </label>
            <input
              id={field}
              type="range"
              min={min}
              max={max}
              step={step}
              value={val}
              onChange={(e) =>
                handleChange(field, parseFloat(e.target.value))
              }
              disabled={loading}
            />
            <div className="form-field__range-labels">
              <span>{min}</span>
              <span>{max}</span>
            </div>
          </div>
        );
      })}

      <button
        type="submit"
        className="submit-btn"
        disabled={loading}
      >
        {loading ? "Analysing…" : "Predict burnout risk"}
      </button>
    </form>
  );
}
