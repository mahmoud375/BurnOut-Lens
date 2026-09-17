import { useState, useRef, useCallback, memo, type FormEvent } from "react";
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

type NumericFieldKey = Exclude<keyof BurnoutRequest, "seniority_level">;

interface SliderFieldProps {
  field: NumericFieldKey;
  label: string;
  min: number;
  max: number;
  step: number;
  initialValue: number;
  disabled: boolean;
  onValueChange: (field: NumericFieldKey, val: number) => void;
}

// Isolated slider component: dragging re-renders ONLY this single field
const SliderField = memo(function SliderField({
  field,
  label,
  min,
  max,
  step,
  initialValue,
  disabled,
  onValueChange,
}: SliderFieldProps) {
  const [val, setVal] = useState<number>(initialValue);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const nextVal = parseFloat(e.target.value);
    setVal(nextVal);
    onValueChange(field, nextVal);
  };

  return (
    <div className="form-field">
      <div className="form-field__label-row">
        <label htmlFor={field}>{label}</label>
        <span className="form-field__value">{val}</span>
      </div>
      <input
        id={field}
        type="range"
        min={min}
        max={max}
        step={step}
        value={val}
        onChange={handleChange}
        disabled={disabled}
      />
      <div className="form-field__range-labels">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
});

export function BurnoutForm({ onResult, loading }: Props) {
  const valuesRef = useRef<BurnoutRequest>({ ...DEFAULTS });
  const [seniority, setSeniority] = useState<SeniorityLevel>(DEFAULTS.seniority_level);

  // Memoized callback so SliderFields never re-render due to parent prop change
  const handleSliderChange = useCallback(
    (field: NumericFieldKey, val: number) => {
      valuesRef.current[field] = val;
    },
    []
  );

  const handleSeniorityChange = useCallback((value: SeniorityLevel) => {
    setSeniority(value);
    valuesRef.current.seniority_level = value;
  }, []);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onResult({ ...valuesRef.current });
  };

  const numericFields = (
    Object.keys(FEATURE_BOUNDS) as Array<NumericFieldKey>
  );

  return (
    <form className="burnout-form" onSubmit={handleSubmit} noValidate>
      <div className="form-header">
        <h2 className="form-title">Work Profile</h2>
        <p className="form-subtitle">Adjust the parameters to simulate your work conditions.</p>
      </div>

      {/* Seniority — select */}
      <div className="form-field">
        <div className="form-field__label-row">
          <label htmlFor="seniority_level">{FEATURE_LABELS.seniority_level}</label>
        </div>
        <div className="select-wrapper">
          <select
            id="seniority_level"
            value={seniority}
            onChange={(e) =>
              handleSeniorityChange(e.target.value as SeniorityLevel)
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
      </div>

      {/* Numeric fields — isolated memoized components */}
      {numericFields.map((field) => {
        const { min, max, step } = FEATURE_BOUNDS[field];
        return (
          <SliderField
            key={field}
            field={field}
            label={FEATURE_LABELS[field]}
            min={min}
            max={max}
            step={step}
            initialValue={DEFAULTS[field]}
            disabled={loading}
            onValueChange={handleSliderChange}
          />
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
