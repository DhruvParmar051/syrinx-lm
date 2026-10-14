"""HellaSwag document processing for the local task: lm-eval's own function, re-exported so the
local YAML (``process_docs: !function utils.process_docs``) runs exactly the stock cleaning."""

from lm_eval.tasks.hellaswag.utils import process_docs

__all__ = ["process_docs"]
