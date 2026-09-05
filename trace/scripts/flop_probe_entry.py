"""torchrun entry point that runs training under a FLOP probe.

Takes exactly the arguments the measured run used, so the probe exercises the
same model, the same loaders and the same gradient scoping.  FLOP_PROBE_UPDATES
sets how many optimizer updates each phase runs before it is cut short.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "implementations", "llmcl_benchmark"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from flop_probe_hooks import install  # noqa: E402

install(int(os.environ.get("FLOP_PROBE_UPDATES", "3")))

from training.main_Ours_LoRA_MoE import main  # noqa: E402

if __name__ == "__main__":
    main()
