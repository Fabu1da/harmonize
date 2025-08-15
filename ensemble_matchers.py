#!/usr/bin/env python
"""
Ensemble matcher classes for combining multiple schema matching approaches.
"""

from typing import Dict, Tuple, List, Union
from collections import Counter
from json_schema import ObjectSchema


class MajorityVoteEnsembleModel:
    """
    Ensemble model that uses majority voting to combine predictions from multiple matchers.
    For each target column, it selects the source column that appears most frequently
    across all matchers, with confidence based on vote count.
    """
    
    def __init__(self, *matchers):
        """
        Initialize with multiple matcher functions/objects.
        
        Args:
            *matchers: Variable number of matcher functions that return dict[str, Tuple[str, float]]
        """
        self.name = "MajorityVote"
        self.matchers = matchers
    
    def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema) -> Dict[str, Tuple[str, float]]:
        """
        Predict column mappings using majority voting.
        
        Args:
            source_schema: Source schema object
            target_schema: Target schema object
            
        Returns:
            Dict mapping target columns to (source_column, confidence) tuples
        """
        # Collect predictions from all matchers
        all_predictions = []
        for matcher in self.matchers:
            if callable(matcher):
                # If it's a function, call it
                predictions = matcher(source_schema, target_schema)
            else:
                # If it's an object with predict method
                predictions = matcher.predict(source_schema, target_schema)
            all_predictions.append(predictions)
        
        # Combine predictions using majority vote
        final_mapping = {}
        
        for target_col in target_schema.properties.keys():
            # Collect all predictions for this target column
            votes = []
            confidences = []
            
            for predictions in all_predictions:
                if target_col in predictions:
                    source_col, conf = predictions[target_col]
                    if source_col and source_col != "—":
                        votes.append(source_col)
                        confidences.append(conf)
            
            if votes:
                # Count votes
                vote_counter = Counter(votes)
                most_common = vote_counter.most_common(1)[0]
                winning_source = most_common[0]
                vote_count = most_common[1]
                
                # Calculate confidence as vote percentage * average confidence of winning votes
                vote_percentage = vote_count / len(votes)
                
                # Get confidences for the winning source column
                winning_confidences = [
                    confidences[i] for i, vote in enumerate(votes) 
                    if vote == winning_source
                ]
                avg_confidence = sum(winning_confidences) / len(winning_confidences)
                
                # Final confidence combines vote strength and matcher confidence
                final_confidence = vote_percentage * avg_confidence
                
                final_mapping[target_col] = (winning_source, final_confidence)
            else:
                # No votes, return no match
                final_mapping[target_col] = ("—", 0.0)
        
        return final_mapping
    




class WeightedScoreEnsembleModel:
    """
    Ensemble model that combines predictions using weighted scoring.
    Each matcher's prediction is weighted by a specified weight, and the
    highest weighted score determines the final prediction.
    """
    
    def __init__(self, *weighted_matchers):
        """
        Initialize with weighted matcher pairs.
        
        Args:
            *weighted_matchers: Variable number of (matcher, weight) tuples
        """
        self.name = "WeightedScore"
        self.weighted_matchers = weighted_matchers
        
        # Normalize weights to sum to 1
        total_weight = sum(weight for _, weight in weighted_matchers)
        self.weighted_matchers = [
            (matcher, weight / total_weight) 
            for matcher, weight in weighted_matchers
        ]
    
    def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema) -> Dict[str, Tuple[str, float]]:
        """
        Predict column mappings using weighted scoring.
        
        Args:
            source_schema: Source schema object
            target_schema: Target schema object
            
        Returns:
            Dict mapping target columns to (source_column, confidence) tuples
        """
        # Collect weighted predictions from all matchers
        all_weighted_predictions = []
        
        for matcher, weight in self.weighted_matchers:
            if callable(matcher):
                # If it's a function, call it
                predictions = matcher(source_schema, target_schema)
            else:
                # If it's an object with predict method
                predictions = matcher.predict(source_schema, target_schema)
            
            all_weighted_predictions.append((predictions, weight))
        
        # Combine predictions using weighted scoring
        final_mapping = {}
        
        for target_col in target_schema.properties.keys():
            # Collect weighted scores for each possible source column
            source_scores = {}
            
            for predictions, weight in all_weighted_predictions:
                if target_col in predictions:
                    source_col, conf = predictions[target_col]
                    if source_col and source_col != "—":
                        weighted_score = conf * weight
                        if source_col in source_scores:
                            source_scores[source_col] += weighted_score
                        else:
                            source_scores[source_col] = weighted_score
            
            if source_scores:
                # Select source column with highest weighted score
                best_source = max(source_scores.keys(), key=lambda k: source_scores[k])
                best_score = source_scores[best_source]
                
                final_mapping[target_col] = (best_source, best_score)
            else:
                # No valid predictions
                final_mapping[target_col] = ("—", 0.0)
        
        return final_mapping


# Helper function to create ensemble matchers with your current setup
def create_ensemble_matchers(gpt_predictions, embed_predictions, cluster_predictions):
    """
    Helper function to create ensemble matchers from existing prediction dictionaries.
    
    Args:
        gpt_predictions: Dict from GPT matcher
        embed_predictions: Dict from embedding matcher  
        cluster_predictions: Dict from clustering matcher
        
    Returns:
        Tuple of (majority_vote_ensemble, weighted_ensemble)
    """
    
    # Create simple matcher functions that return the pre-computed predictions
    def gpt_matcher(source_schema, target_schema):
        return gpt_predictions
    
    def embed_matcher(source_schema, target_schema):
        return embed_predictions
    
    def cluster_matcher(source_schema, target_schema):
        return cluster_predictions
    
    # Create ensemble models
    majority_ensemble = MajorityVoteEnsembleModel(
        gpt_matcher, embed_matcher, cluster_matcher
    )
    
    weighted_ensemble = WeightedScoreEnsembleModel(
        (gpt_matcher, 0.5),
        (embed_matcher, 0.3), 
        (cluster_matcher, 0.2),
    )
    
    return majority_ensemble, weighted_ensemble
