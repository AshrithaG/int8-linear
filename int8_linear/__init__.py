"""W8A8 int8 linear layers for LLM inference on consumer GPUs, written in Triton."""

from int8_linear.kernel import Config, NotSupported, default_config, tuned_config, w8a8_mm

__all__ = ["Config", "NotSupported", "default_config", "tuned_config", "w8a8_mm"]
