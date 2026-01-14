"""
TumorImagingBench Models Module

This module contains various foundation model feature extractors.
Each extractor is imported conditionally based on available dependencies.

Key Features:
- Graceful degradation for missing dependencies
- Dynamic model registration via register_extractor()
- Type validation for custom models
"""

from .base import BaseModel

# Dictionary to track available extractors
AVAILABLE_EXTRACTORS = {}


def register_extractor(name, extractor_class):
    """
    Register a custom model extractor.

    This function allows users to register custom foundation models at runtime.
    Custom models must inherit from BaseModel and implement the required
    abstract methods: load(), preprocess(), and forward().

    Parameters:
    -----------
    name : str
        Unique identifier for the extractor (used in get_extractor())
    extractor_class : type
        A class that inherits from BaseModel

    Raises:
    -------
    TypeError
        If extractor_class does not inherit from BaseModel

    Examples:
    ---------
    >>> from tumorimagingbench.models import register_extractor
    >>> from my_models import MyCustomExtractor
    >>> register_extractor('MyCustomExtractor', MyCustomExtractor)
    >>>
    >>> # Now use it with the standard interface
    >>> from tumorimagingbench.models import get_extractor
    >>> model_class = get_extractor('MyCustomExtractor')
    """
    # Validate that the extractor inherits from BaseModel
    if not issubclass(extractor_class, BaseModel):
        raise TypeError(
            f"Extractor '{name}' must inherit from BaseModel. "
            f"Got {extractor_class.__name__} which inherits from "
            f"{[base.__name__ for base in extractor_class.__bases__]}"
        )

    AVAILABLE_EXTRACTORS[name] = extractor_class
    print(f"✓ Registered extractor: {name}")


def get_available_extractors():
    """
    Return a list of available extractor names.

    Returns:
    --------
    list
        List of registered extractor names (strings)

    Examples:
    ---------
    >>> from tumorimagingbench.models import get_available_extractors
    >>> extractors = get_available_extractors()
    >>> print(f"Available extractors: {extractors}")
    Available extractors: ['CTClipVitExtractor', ...]
    """
    return list(AVAILABLE_EXTRACTORS.keys())


def get_extractor(name):
    """
    Get an extractor class by name.

    Retrieves a registered extractor class for instantiation.

    Parameters:
    -----------
    name : str
        Name of the extractor (must be registered via register_extractor() or default imports)

    Returns:
    --------
    type
        The extractor class

    Raises:
    -------
    ValueError
        If the extractor is not registered

    Examples:
    ---------
    >>> from tumorimagingbench.models import get_extractor
    >>> CTClipVitExtractor = get_extractor('CTClipVitExtractor')
    >>> model = CTClipVitExtractor()
    """
    if name in AVAILABLE_EXTRACTORS:
        return AVAILABLE_EXTRACTORS[name]
    else:
        available = get_available_extractors()
        raise ValueError(
            f"Extractor '{name}' is not available.\n"
            f"Available extractors: {available}"
        )


# Try to import each default extractor, handling missing dependencies gracefully
try:
    from .ct_clip_vit import CTClipVitExtractor
    register_extractor('CTClipVitExtractor', CTClipVitExtractor)
except ImportError as e:
    print(f"Warning: CTClipVitExtractor not available due to missing dependencies: {e}")

try:
    from .ct_fm import CTFMExtractor
    register_extractor('CTFMExtractor', CTFMExtractor)
except ImportError as e:
    print(f"Warning: CTFMExtractor not available due to missing dependencies: {e}")

try:
    from .fmcib import FMCIBExtractor
    register_extractor('FMCIBExtractor', FMCIBExtractor)
except ImportError as e:
    print(f"Warning: FMCIBExtractor not available due to missing dependencies: {e}")

try:
    from .medimageinsight import MedImageInsightExtractor
    register_extractor('MedImageInsightExtractor', MedImageInsightExtractor)
except ImportError as e:
    print(f"Warning: MedImageInsightExtractor not available due to missing dependencies: {e}")

try:
    from .merlin import MerlinExtractor
    register_extractor('MerlinExtractor', MerlinExtractor)
except ImportError as e:
    print(f"Warning: MerlinExtractor not available due to missing dependencies: {e}")

try:
    from .modelsgen import ModelsGenExtractor
    register_extractor('ModelsGenExtractor', ModelsGenExtractor)
except ImportError as e:
    print(f"Warning: ModelsGenExtractor not available due to missing dependencies: {e}")

try:
    from .pasta import PASTAExtractor
    register_extractor('PASTAExtractor', PASTAExtractor)
except ImportError as e:
    print(f"Warning: PASTAExtractor not available due to missing dependencies: {e}")

try:
    from .suprem import SUPREMExtractor
    register_extractor('SUPREMExtractor', SUPREMExtractor)
except ImportError as e:
    print(f"Warning: SUPREMExtractor not available due to missing dependencies: {e}")

try:
    from .vista3d import VISTA3DExtractor
    register_extractor('VISTA3DExtractor', VISTA3DExtractor)
except ImportError as e:
    print(f"Warning: VISTA3DExtractor not available due to missing dependencies: {e}")

try:
    from .voco import VocoExtractor
    register_extractor('VocoExtractor', VocoExtractor)
except ImportError as e:
    print(f"Warning: VocoExtractor not available due to missing dependencies: {e}")


# Make available extractors accessible at module level
__all__ = [
    'AVAILABLE_EXTRACTORS',
    'get_available_extractors',
    'get_extractor',
    'register_extractor',
    'BaseModel'
] + list(AVAILABLE_EXTRACTORS.keys())