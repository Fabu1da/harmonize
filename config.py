from approaches.base import BaseApproach
from approaches.cluster import ClusteringApproach
from approaches.coma import ComaApproach
from approaches.edit_distance import EditDistanceApproach
from approaches.embedding import EmbeddingApproach
from approaches.ensemble import WeightedEnsembleApproach, MajorityVoteEnsembleApproach
from approaches.identity import IdentityApproach
from approaches.llm import LLMApproach
from approaches.null import NullApproach
from approaches.random import RandomApproach

APPROACHES: list[tuple[BaseApproach, str]] = [
    (RandomApproach(), "Random"),
    (NullApproach(), "Null"),
    (IdentityApproach(), "Identity"),
    (EditDistanceApproach(threshold=0.1), "Edit Distance"),

    (ComaApproach(use_instances=False), "COMA (without instances)"),
    (ComaApproach(use_instances=True), "COMA (with instances)"),

    (LLMApproach(model="gpt-5-nano"), "LLM (GPT-5 nano)"),
    (LLMApproach(model="gpt-5-mini"), "LLM (GPT-5 mini)"),

    (ClusteringApproach(model="text-embedding-3-large"), "Clustering (Text Embedding Large)"),
    (EmbeddingApproach(model="text-embedding-3-large", threshold=0.6, method="greedy"), "Embedding (Text Embedding Large) (Greedy)"),
    (EmbeddingApproach(model="text-embedding-3-small", threshold=0.6, method="hungarian"), "Embedding (Text Embedding Small) (Hungarian)"),
    (EmbeddingApproach(model="text-embedding-3-large", threshold=0.6, method="hungarian"), "Embedding (Text Embedding Large) (Hungarian)"),

    (MajorityVoteEnsembleApproach([
        "LLM (GPT-5 mini)",
        "Embedding (Text Embedding Large) (Hungarian)",
        "Edit Distance",
    ]), "Majority Vote Ensemble (GPT-5 mini, Text Embedding Large, Edit Distance)"),
     (WeightedEnsembleApproach([
        "LLM (GPT-5 mini)",
        "Edit Distance",
        "Embedding (Text Embedding Large) (Hungarian)",
    ], weights=[0.45, 0.3, 0.25], use_confidence=True), "Weighted Ensemble (GPT-5 mini, Edit Distance, Text Embedding Large)"),
     (WeightedEnsembleApproach([
        "calibrated/LLM (GPT-5 mini)",
        "calibrated/Edit Distance",
        "calibrated/Embedding (Text Embedding Large) (Hungarian)",
    ], weights=[0.45, 0.3, 0.25], use_confidence=True), "Calibrated Weighted Ensemble (GPT-5 mini, Edit Distance, Text Embedding Large)"),
]
APPROACH_NAMES = [name for _, name in APPROACHES]
