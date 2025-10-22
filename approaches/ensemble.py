from typing import Optional

from .base import BaseApproach
from json_schema import ObjectSchema

class BaseEnsembleApproach(BaseApproach):
    components: list[BaseApproach]

    def __init__(self, components: list[BaseApproach]):
        self.components = components

    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        predictions = []
        for component in self.components:
            pred = await component.predict(source_schema, target_schema, **kwargs)
            predictions.append(pred)
        
        combined = {}
        target_columns = list(target_schema.properties.keys())
        for target_col in target_columns:
            candidates = [pred.get(target_col, (None, 0.0, "No prediction")) for pred in predictions]
            combined[target_col] = self.aggregate(candidates)
        return combined

    def aggregate(self, candidates: list[tuple[Optional[str], float, Optional[str]]]) -> tuple[Optional[str], float, Optional[str]]:
        raise NotImplementedError("Subclasses should implement this method.")

class WeightedEnsembleApproach(BaseEnsembleApproach):
    weights: list[float]
    use_confidence: bool

    def __init__(self, components: list[BaseApproach], weights: list[float], use_confidence: bool):
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
        confidence = vote_count[best_match] / sum(self.weights)
        reasons = [reason for src, conf, reason in candidates if src == best_match and reason]
        combined_reason = "; ".join(reasons) if reasons else None
        return best_match, confidence, combined_reason

class MajorityVoteEnsembleApproach(WeightedEnsembleApproach):
    def __init__(self, components: list[BaseApproach]):
        weights = [1.0] * len(components)
        super().__init__(components, weights, use_confidence=False)
