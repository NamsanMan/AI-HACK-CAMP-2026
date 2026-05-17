/** FINAL_TEST.md / export 스펙 — fake_probability 기준 UI 히스테리시스 */

let riskState = "Low";

export function resetRiskHysteresis() {
  riskState = "Low";
}

/**
 * @param {number} fakeProbability 0..1
 * @returns {"Low"|"Watch"|"High"}
 */
export function updateRiskHysteresis(fakeProbability) {
  const s = Math.min(1, Math.max(0, fakeProbability));

  if (riskState === "Low") {
    if (s >= 0.65) {
      riskState = "High";
    } else if (s >= 0.55) {
      riskState = "Watch";
    }
  } else if (riskState === "Watch") {
    if (s >= 0.65) {
      riskState = "High";
    } else if (s < 0.45) {
      riskState = "Low";
    }
  } else if (riskState === "High") {
    if (s < 0.55) {
      riskState = s < 0.45 ? "Low" : "Watch";
    }
  }

  return riskState;
}

export function getRiskHysteresisState() {
  return riskState;
}

export function riskStateToLevel(state) {
  if (state === "High") {
    return "warning";
  }

  if (state === "Watch") {
    return "caution";
  }

  return "normal";
}
