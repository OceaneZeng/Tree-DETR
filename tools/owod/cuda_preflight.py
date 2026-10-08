"""Check CUDA support without starting training."""
import subprocess
import sys

def validate_cuda_runtime(environment):
    """Fail before the queue if the active torch build cannot run the GPUs."""
    probe = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        "arch=set(torch.cuda.get_arch_list()); "
        "caps=[torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]; "
        "missing=[(i, f'sm_{m}{n}') for i,(m,n) in enumerate(caps) if arch and f'sm_{m}{n}' not in arch]; "
        "\nif missing:\n"
        "    raise SystemExit('unsupported GPU architectures: ' + repr(missing) + '; torch=' + torch.__version__)\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], env=environment,
                            text=True, capture_output=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "CUDA runtime preflight failed")
