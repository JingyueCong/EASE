import importlib
from typing import Dict, Any
from omegaconf import DictConfig
from evals.tofu import TOFUEvaluator
from evals.muse import MUSEEvaluator

EVALUATOR_REGISTRY: Dict[str, Any] = {}
OPTIONAL_EVALUATORS = {
    "LMEvalEvaluator": ("evals.lm_eval", "LMEvalEvaluator", "lm-eval"),
}


def _register_evaluator(evaluator_class):
    EVALUATOR_REGISTRY[evaluator_class.__name__] = evaluator_class


def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator_handler_name = eval_cfg.get("handler")
    assert evaluator_handler_name is not None, ValueError(f"{name} handler not set")
    eval_handler = EVALUATOR_REGISTRY.get(evaluator_handler_name)
    if eval_handler is None and evaluator_handler_name in OPTIONAL_EVALUATORS:
        module_name, class_name, dependency = OPTIONAL_EVALUATORS[evaluator_handler_name]
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"{evaluator_handler_name} requires the optional `{dependency}` "
                "dependency, but that evaluator was selected and the dependency "
                "is not installed."
            ) from exc
        eval_handler = getattr(module, class_name)
        _register_evaluator(eval_handler)
    if eval_handler is None:
        raise NotImplementedError(
            f"{evaluator_handler_name} not implemented or not registered"
        )
    return eval_handler(eval_cfg, **kwargs)


def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators


# Register Your benchmark evaluators
_register_evaluator(TOFUEvaluator)
_register_evaluator(MUSEEvaluator)
