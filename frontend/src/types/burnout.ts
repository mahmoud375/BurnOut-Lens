/** TypeScript interfaces that exactly mirror backend/app/schemas/burnout.py */

// ── Request ─────────────────────────────────────────────────────────────────

export type SeniorityLevel =
  | "Junior"
  | "Mid"
  | "Senior"
  | "Lead"
  | "Manager"
  | "Principal";

export interface BurnoutRequest {
  seniority_level: SeniorityLevel;
  work_hours_per_week: number;        // ge=20, le=90
  meetings_per_day: number;           // ge=0,  le=15
  sleep_hours_per_night: number;      // ge=2,  le=12
  exercise_days_per_week: number;     // ge=0,  le=7
  vacation_days_taken: number;        // ge=0,  le=30
  social_support_score: number;       // ge=1,  le=10
  manager_support_score: number;      // ge=1,  le=10
  deadline_pressure_score: number;    // ge=1,  le=10
}

// Field bounds matching config.FEATURE_BOUNDS — single source of truth for
// HTML min/max attributes (avoids hardcoding the same numbers twice).
export const FEATURE_BOUNDS: Record<
  Exclude<keyof BurnoutRequest, "seniority_level">,
  { min: number; max: number; step: number }
> = {
  work_hours_per_week:    { min: 20,  max: 90,  step: 1   },
  meetings_per_day:       { min: 0,   max: 15,  step: 1   },
  sleep_hours_per_night:  { min: 2,   max: 12,  step: 0.5 },
  exercise_days_per_week: { min: 0,   max: 7,   step: 1   },
  vacation_days_taken:    { min: 0,   max: 30,  step: 1   },
  social_support_score:   { min: 1,   max: 10,  step: 0.5 },
  manager_support_score:  { min: 1,   max: 10,  step: 0.5 },
  deadline_pressure_score:{ min: 1,   max: 10,  step: 0.5 },
};

// Human-readable labels — shared between BurnoutForm and ResultPanel
export const FEATURE_LABELS: Record<keyof BurnoutRequest, string> = {
  seniority_level:          "Seniority level",
  work_hours_per_week:      "Work hours per week",
  meetings_per_day:         "Meetings per day",
  sleep_hours_per_night:    "Sleep hours per night",
  exercise_days_per_week:   "Exercise days per week",
  vacation_days_taken:      "Vacation days taken",
  social_support_score:     "Social support (1–10)",
  manager_support_score:    "Manager support (1–10)",
  deadline_pressure_score:  "Deadline pressure (1–10)",
};

export const SENIORITY_LEVELS: SeniorityLevel[] = [
  "Junior", "Mid", "Senior", "Lead", "Manager", "Principal",
];

// ── Response ─────────────────────────────────────────────────────────────────

export type BurnoutLevel = "Low" | "Moderate" | "High" | "Severe";

export interface Contribution {
  feature: string;
  value: number;
  shap_value: number;
  direction: "increases" | "decreases";
}

export interface BurnoutResponse {
  burnout_score: number;
  burnout_level: BurnoutLevel;
  base_value: number;
  contributions: Contribution[];
}
