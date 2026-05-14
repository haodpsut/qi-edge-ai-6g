"""Model registry. All models target ~3.6K params for fair comparison."""
from .mlp import MLP
from .cnn1d import CNN1D
from .efficient_kan import EfficientKAN
from .fourier_kan import FourierKAN
from .complex_mlp import ComplexMLP

MODELS = {
    "mlp": MLP,
    "cnn1d": CNN1D,
    "kan": EfficientKAN,
    "fkan": FourierKAN,
    "cmlp": ComplexMLP,
}

__all__ = ["MODELS", "MLP", "CNN1D", "EfficientKAN", "FourierKAN", "ComplexMLP"]