"""Loaded at interpreter start when this directory is on PYTHONPATH.

Does nothing unless KVP_PROTECT or KVP_KEEPALIVE is set. Patches are applied
lazily, right after vLLM's scheduler module is imported, so unrelated Python
processes (compile workers, tokenizer helpers) never import vLLM because of us.
"""
import os

if os.environ.get("KVP_PROTECT") == "1" or os.environ.get("KVP_KEEPALIVE") == "1":
    import importlib.abc
    import sys

    _TARGET = "vllm.v1.core.sched.scheduler"

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            import importlib.util
            spec = importlib.util.find_spec(name)
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module):
                orig_exec(module)
                import kvp_patch
                kvp_patch.apply()

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())
