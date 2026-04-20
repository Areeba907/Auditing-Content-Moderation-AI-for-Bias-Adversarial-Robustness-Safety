
import re
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BLOCKLIST = {
    "direct_threat": [
        re.compile(r"\bi(?:'ll| will| am going to| gonna)\s+(kill|hurt|shoot|stab|murder)\s+you\b", re.IGNORECASE),
        re.compile(r"\byou(?:'re| are)\s+going\s+to\s+die\b", re.IGNORECASE),
        re.compile(r"\bsomeone\s+should\s+(kill|shoot|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\bi(?:'ll| will)\s+find\s+where\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\bi(?:'ll| will)\s+come\s+for\s+you\b", re.IGNORECASE),
    ],
    "self_harm_directed": [
        re.compile(r"\bgo\s+kill\s+yourself\b", re.IGNORECASE),
        re.compile(r"\byou\s+should\s+kill\s+yourself\b", re.IGNORECASE),
        re.compile(r"\bnobody\s+would\s+miss\s+you\s+if\s+you\s+died\b", re.IGNORECASE),
        re.compile(r"\bdo\s+everyone\s+a\s+favor\s+and\s+disappear\b", re.IGNORECASE),
    ],
    "doxxing_stalking": [
        re.compile(r"\bi\s+know\s+where\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\bi(?:'ll| will)\s+post\s+your\s+address\b", re.IGNORECASE),
        re.compile(r"\bi\s+found\s+your\s+real\s+name\b", re.IGNORECASE),
        re.compile(r"\beveryone\s+will\s+know\s+who\s+you\s+really\s+are\b", re.IGNORECASE),
    ],
    "dehumanization": [
        re.compile(r"\b\w+\s+are\s+not\s+(?:human|people|persons)\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+are\s+animals\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+should\s+be\s+exterminated\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+are\s+a\s+disease\b", re.IGNORECASE),
    ],
    "coordinated_harassment": [
        re.compile(r"\beveryone\s+report\s+\w+\b", re.IGNORECASE),
        re.compile(r"\blet(?:'s|\s+us)\s+all\s+go\s+after\b", re.IGNORECASE),
        re.compile(r"\braid\s+their\s+profile\b", re.IGNORECASE),
        re.compile(r"\bmass\s+report\b(?=.*account)", re.IGNORECASE),
    ],
}

def input_filter(text: str):
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0
                }
    return None


class ModerationPipeline:
    def __init__(
        self,
        model_dir="saved_models/part4_oversample_model",
        calibrator_path="saved_models/part5_isotonic_calibrator.pkl",
        threshold_allow=0.4,
        threshold_block=0.6,
        max_len=128
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = model_dir
        self.calibrator_path = calibrator_path
        self.threshold_allow = threshold_allow
        self.threshold_block = threshold_block
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

        self.calibrator = joblib.load(calibrator_path)

    def predict_raw_proba(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits
            prob = torch.softmax(logits, dim=1)[:, 1].item()

        return float(prob)

    def predict_calibrated_proba(self, text: str) -> float:
        raw_prob = self.predict_raw_proba(text)
        calibrated_prob = float(self.calibrator.predict([raw_prob])[0])
        return calibrated_prob

    def predict(self, text: str):
        layer1 = input_filter(text)
        if layer1 is not None:
            return layer1

        calibrated_prob = self.predict_calibrated_proba(text)

        if calibrated_prob >= self.threshold_block:
            return {
                "decision": "block",
                "layer": "model",
                "confidence": calibrated_prob
            }
        elif calibrated_prob <= self.threshold_allow:
            return {
                "decision": "allow",
                "layer": "model",
                "confidence": calibrated_prob
            }
        else:
            return {
                "decision": "review",
                "layer": "model",
                "confidence": calibrated_prob
            }
