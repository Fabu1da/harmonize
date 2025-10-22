from approaches.base import BaseApproach
from approaches.coma import ComaApproach
from approaches.llm import LLMApproach
from approaches.embedding import EmbeddingApproach
from approaches.cluster import ClusteringApproach
from approaches.random import RandomApproach
from approaches.ensemble import WeightedEnsembleApproach, MajorityVoteEnsembleApproach

APPROACHES: list[tuple[BaseApproach, str]] = [
    (RandomApproach(), "Random"),
    (LLMApproach(model="gpt-5-mini"), "LLM (GPT-5 mini)"),
    (LLMApproach(model="gpt-4o-mini"), "LLM (GPT-4o mini)"),
    (ComaApproach(), "COMA"),
    (EmbeddingApproach(model="text-embedding-ada-002", threshold=0.6, method="hungarian"), "Embedding (Ada 002) (Hungarian)"),
    (EmbeddingApproach(model="text-embedding-ada-002", threshold=0.6, method="greedy"), "Embedding (Ada 002) (Greedy)"),
    (ClusteringApproach(model="text-embedding-ada-002"), "Clustering"),
    
    (WeightedEnsembleApproach([
        LLMApproach(model="gpt-4o-mini"),
        EmbeddingApproach(model="text-embedding-ada-002", threshold=0.6, method="greedy"),
        ClusteringApproach(model="text-embedding-ada-002"),
    ], weights=[0.5, 0.2, 0.3], use_confidence=True), "Ensemble (GPT-4o mini, Ada 002, Clustering)"),
    (MajorityVoteEnsembleApproach([
        LLMApproach(model="gpt-4o-mini"),
        EmbeddingApproach(model="text-embedding-ada-002", threshold=0.6, method="greedy"),
        ClusteringApproach(model="text-embedding-ada-002"),
    ]), "Majority Vote Ensemble (GPT-4o mini, Ada 002, Clustering)"),
]
