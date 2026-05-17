import {
  getRiskHysteresisState,
  resetRiskHysteresis,
  riskStateToLevel,
  updateRiskHysteresis,
} from "./riskHysteresis.js";

export { resetRiskHysteresis };

const LABELS = {
  Low: { title: "낮은 위험", message: "현재 영상에서는 큰 이상 신호가 감지되지 않았습니다." },
  Watch: {
    title: "관찰",
    message: "딥페이크 위험 점수가 관찰 구간입니다. 추가 확인을 권장합니다.",
  },
  High: {
    title: "높은 위험",
    message: "딥페이크 위험 점수가 높습니다. HR 표시는 신뢰할 수 없습니다.",
  },
};

/**
 * fake_probability + 히스테리시스 기준 UI 상태.
 * confidence_score는 내부 참고만 (표시 최소화).
 */
export function getRiskStatus(result, ready) {
  if (!ready || !result) {
    return {
      riskLevel: "warming",
      riskState: null,
      trustLevel: "not_ready",
      title: ready ? "점수 계산 중" : "분석 준비 중",
      message:
        ready && !result
          ? "rPPG 버퍼가 채워졌습니다. 곧 위험·생동성 점수가 표시됩니다."
          : "rPPG 신호를 모으는 중입니다. (얼굴이 화면에 보여야 합니다)",
    };
  }

  const fakeProb = result.fakeProb;
  const riskState = updateRiskHysteresis(fakeProb);
  const riskLevel = riskStateToLevel(riskState);
  const labels = LABELS[riskState] ?? LABELS.Low;

  let message = labels.message;

  if (result.livenessScore < 0.4 && fakeProb >= 0.55) {
    message += ` (생동성 ${result.livenessScore.toFixed(2)})`;
  }

  if (result.confidence < 0.45) {
    message += " 신호 품질이 낮아 판정 신뢰도가 제한될 수 있습니다.";
  }

  return {
    riskLevel,
    riskState,
    trustLevel: result.confidence >= 0.45 ? "reliable" : "uncertain",
    title: labels.title,
    message,
  };
}

export function shouldShowHr(riskState) {
  return riskState !== "High";
}

export function formatHr(hrPred, riskState) {
  if (!shouldShowHr(riskState)) {
    return "HR: 신뢰 불가";
  }

  if (typeof hrPred !== "number" || hrPred <= 0 || Number.isNaN(hrPred)) {
    return "HR: —";
  }

  return `HR: ${Math.round(hrPred)} bpm`;
}
