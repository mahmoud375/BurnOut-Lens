import type { BurnoutRequest, BurnoutResponse } from "../types/burnout";

// Read backend URL from Vite env — set VITE_API_BASE_URL in .env for
// non-localhost deployments (Docker, staging, prod).
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/**
 * POST /predict — returns a burnout prediction with SHAP explanation.
 *
 * Throws an Error with a human-readable message on any non-2xx response.
 * For 422 validation errors, the FastAPI detail array is included so the
 * caller can surface which field(s) failed.
 */
export async function getBurnoutPrediction(
  data: BurnoutRequest
): Promise<BurnoutResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    // 422: Pydantic validation error — detail is an array of field errors
    // Other errors: detail may be a string or absent
    let message = `Request failed (${response.status})`;
    try {
      const err = await response.json();
      if (Array.isArray(err.detail)) {
        // Pick the first validation error's human message
        message = err.detail.map((d: { msg: string }) => d.msg).join("; ");
      } else if (typeof err.detail === "string") {
        message = err.detail;
      }
    } catch {
      // response body not JSON — keep the generic message
    }
    throw new Error(message);
  }

  return response.json() as Promise<BurnoutResponse>;
}
