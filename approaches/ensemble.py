import json
from typing import Optional

from .base import BaseApproach
from json_schema import ObjectSchema

class BaseEnsembleApproach(BaseApproach):
    components: list[str]

    def __init__(self, components: list[str]):
        self.components = components

    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        assert target_schema.properties is not None

        expected_name: str = kwargs["expected_name"]

        predictions: list[dict[str, tuple[Optional[str], float, Optional[str]]]] = []

        for component in self.components:
            predicted_path = f"./assets/predicted/{component}/{expected_name}.json"

            with open(predicted_path) as f:
                pred = json.load(f)

            predictions.append(pred)
        
        combined = {}
        target_columns = list(target_schema.properties.keys())

        for target_col in target_columns:
            candidates = [pred[target_col] for pred in predictions]
            combined[target_col] = self.aggregate(candidates)

        return combined

    def aggregate(self, candidates: list[tuple[Optional[str], float, Optional[str]]]) -> tuple[Optional[str], float, Optional[str]]:
        raise NotImplementedError("Subclasses should implement this method.")

class WeightedEnsembleApproach(BaseEnsembleApproach):
    weights: list[float]
    use_confidence: bool

    def __init__(self, components: list[str], weights: list[float], use_confidence: bool):
        super().__init__(components)
        self.weights = weights
        self.use_confidence = use_confidence
        assert len(components) == len(weights), "Number of components must match number of weights"

    def aggregate(self, candidates: list[tuple[Optional[str], float, Optional[str]]]) -> tuple[Optional[str], float, Optional[str]]:
        vote_count = {}
        for i, (source_col, confidence, reasoning) in enumerate(candidates):
            if source_col not in vote_count:
                vote_count[source_col] = 0
            if self.use_confidence:
                vote_count[source_col] += confidence * self.weights[i]
            else:
                vote_count[source_col] += self.weights[i]
        best_match = max(vote_count.items(), key=lambda x: x[1])[0]
        vote_sum = sum(vote_count.values())
        confidence = vote_count[best_match] / vote_sum if vote_sum > 0 else 0
        reasons = [reason for src, conf, reason in candidates if src == best_match and reason]
        combined_reason = "; ".join(reasons) if reasons else None
        return best_match, confidence, combined_reason

class MajorityVoteEnsembleApproach(WeightedEnsembleApproach):
    def __init__(self, components: list[str]):
        weights = [1.0] * len(components)
        super().__init__(components, weights, use_confidence=False)
