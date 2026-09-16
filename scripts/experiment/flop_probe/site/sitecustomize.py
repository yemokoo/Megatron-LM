"""FLOP-probe import hook (opt-in).

Only active when FLOP_PROBE_STEPS is set in the environment. Put this
directory on PYTHONPATH and every python process (torchrun workers included)
will, upon importing ``megatron.training.training``, wrap ``train_step`` with a
FlopCounterMode probe.  Nothing in the repo is modified; production launches
without FLOP_PROBE_STEPS are untouched (the hook is never registered).
"""
import os
import sys

if os.environ.get("FLOP_PROBE_STEPS"):
    import importlib.abc
    import importlib.util

    _TARGET = "megatron.training.training"

    class _PostImportHook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != _TARGET:
                return None
            # Delegate to the normal finders (remove self to avoid recursion).
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                if self not in sys.meta_path:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec, _self=self):
                _orig(module)
                if _self in sys.meta_path:
                    sys.meta_path.remove(_self)  # one-shot
                here = os.path.dirname(os.path.abspath(__file__))
                parent = os.path.dirname(here)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                import flop_probe_megatron
                flop_probe_megatron.install(module)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PostImportHook())
