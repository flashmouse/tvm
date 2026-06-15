eval "$(micromamba shell hook --shell bash)"
micromamba activate tvm-build-venv

mkdir -p build
cp cmake/config.cmake build/config.cmake

python - <<'PY'
from pathlib import Path
import re

path = Path("build/config.cmake")
text = path.read_text()

replacements = {
    r"set\(CMAKE_BUILD_TYPE .*?\)": "set(CMAKE_BUILD_TYPE RelWithDebInfo)",
    r"set\(USE_LLVM .*?\)": 'set(USE_LLVM "$ENV{CONDA_PREFIX}/bin/llvm-config")',
    r"set\(HIDE_PRIVATE_SYMBOLS .*?\)": "set(HIDE_PRIVATE_SYMBOLS ON)",
    r"set\(USE_CUDA .*?\)": "set(USE_CUDA OFF)",
    r"set\(USE_METAL .*?\)": "set(USE_METAL OFF)",
    r"set\(USE_VULKAN .*?\)": "set(USE_VULKAN OFF)",
    r"set\(USE_OPENCL .*?\)": "set(USE_OPENCL OFF)",
    r"set\(USE_CUBLAS .*?\)": "set(USE_CUBLAS OFF)",
    r"set\(USE_CUDNN .*?\)": "set(USE_CUDNN OFF)",
    r"set\(USE_CUTLASS .*?\)": "set(USE_CUTLASS OFF)",
}

for pattern, replacement in replacements.items():
    text, count = re.subn(pattern, replacement, text, count=1)
    if count == 0:
        text += f"\n{replacement}\n"

path.write_text(text)
PY

export TVM_LIBRARY_PATH="$(pwd)/build/lib"
export DYLD_LIBRARY_PATH="$(pwd)/build/lib"
# Pin the Python sources to this checkout so they match the lib above
# (overrides any stale editable install pointing at another worktree).
export PYTHONPATH="$(pwd)/python${PYTHONPATH:+:$PYTHONPATH}"
