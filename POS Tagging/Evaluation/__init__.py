# evaluation/__init__.py
from .metrics import (
    analyze_errors,
    calculate_metrics
)

from .visualization import (
    plot_confusion_matrix,
    plot_comparison_chart
)

__all__ = [
    'analyze_errors',
    'calculate_metrics',
    'plot_confusion_matrix',
    'plot_comparison_chart'
]