"""Execute the demo notebook with the frozen kernel copies (next/kernel/stable) first on the import path."""
import sys, os, runpy
sys.path.insert(0, "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable")
os.environ["PYTHONPATH"] = "/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/stable:" + os.environ.get("PYTHONPATH", "")
import stream_cqsa.interface as I
print("kernels:", I.cqsa_cuda.__file__, I.cqsa_cuda_noncausal.__file__, flush=True)
os.chdir("/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/notebooks")
sys.argv = ["make_oom_demo.py"]
runpy.run_path("make_oom_demo.py", run_name="__main__")
