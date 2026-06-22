"""Aggregate verbalized confidence + trajectory features → trajectory success probability."""
from compass.trajectory import TrajectoryFeatures


def aggregate(verbalized_confidence: float, features: TrajectoryFeatures) -> float:
    """Hand-tuned aggregator; locked on 5-task dev split before eval."""
    raise NotImplementedError
