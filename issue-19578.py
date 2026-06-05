import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as R
from tvm.relax.transform import LegalizeOps

bb = relax.BlockBuilder()
a = relax.Var("a", relax.TensorStructInfo((4, 0), "float32"))
b = relax.Var("b", relax.TensorStructInfo((0, 4), "float32"))
with bb.function("main", [a, b]):
    with bb.dataflow():
        gv = bb.emit_output(bb.emit(R.matmul(a, b)))
    bb.emit_func_output(gv)
mod = bb.finalize()


from tvm.ir.instrument import PrintBeforeAll, PrintAfterAll

# pipeline = tvm.ir.transform.Sequential([LegalizeOps()])
# with tvm.ir.transform.PassContext(instruments=[PrintBeforeAll(), PrintAfterAll()]):
#     legalized = pipeline(mod)          # ← 在块内调用,才会触发 instrument
# exe = tvm.relax.build(legalized, target="llvm")

pipeline = tvm.ir.transform.Sequential([LegalizeOps()])
exe = tvm.relax.build(pipeline(mod), target="llvm")
vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())

out = vm["main"](
    tvm.runtime.tensor(np.zeros((4, 0), np.float32), device=tvm.cpu()),
    tvm.runtime.tensor(np.zeros((0, 4), np.float32), device=tvm.cpu()),
).numpy()

print(out)
# Non-deterministic garbage, e.g.:
# [[ 6.19e+21  3.07e-41  5.25e+21  3.07e-41]
#  [ 6.19e+21  3.07e-41  5.25e+21  3.07e-41]
#  [       nan -7.43e-44        nan -1.40e-44]
#  [       nan  7.57e-44        nan  4.48e-44]]

print(np.zeros((4, 0)) @ np.zeros((0, 4)))
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
