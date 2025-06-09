"""
Dataset Sources Package
Individual dataset source implementations
"""

try:
    from .wikipedia import WikipediaSource
except ImportError:
    WikipediaSource = None

try:
    from .trec import TRECSource
except ImportError:
    TRECSource = None
    
try:
    from .miracl import MIRACLSource  
except ImportError:
    MIRACLSource = None
    
try:
    from .live_apis import LiveAPISource
except ImportError:
    LiveAPISource = None

__all__ = ["WikipediaSource", "TRECSource", "MIRACLSource", "LiveAPISource"]