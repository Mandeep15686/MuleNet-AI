"""
MuleNet AI — Graph-based money-mule detection on the IBM AML HI-Small dataset.

Modules
-------
train       Feature engineering + FeatureOnlyGNN training (primary pipeline)
explain     Reason-code post-processor for flagged accounts
rings       Louvain community ring detection
bootstrap   Paired bootstrap comparison between two model versions
dashboard   HTML investigation dashboard builder
"""

__version__ = "2.9.2"
__all__ = ["train", "explain", "rings", "bootstrap", "dashboard"]
