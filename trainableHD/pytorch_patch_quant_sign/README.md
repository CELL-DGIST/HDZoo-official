# PyTorch patch to support sign() for quantized data type
This patch is designed for PyTorch version 1.20.0 and specifically targets the `aten/src/ATen/native/quantized/` directory in the PyTorch codebase.
This patch includes all the files from the specific version's directory in the PyTorch codebase for your convenience, not just the parts that have been modified.
Thus, except for the patched parts, all related licenses and rights are as per the [original PyTorch repository](https://github.com/pytorch/pytorch).

### How to Apply the Patch:
1. **For PyTorch 1.20.0 Users:**
   - Directly overwrite the `aten/src/ATen/native/quantized/` directory in your PyTorch source code with the provided patch.

2. **For Other Versions:**
   - To ensure a safe build, locate the comments marked with "Yeseong" in your PyTorch source code. You can use the command `grep -nr 'Yeseong' .` to find these comments.
   - Apply the changes indicated by these comments to your version of the PyTorch source code.

### You should build the pytorch code with the patched file
- Please refer to `Installation` -> `From Source` section in the [original PyTorch repository](https://github.com/pytorch/pytorch).

### Files Requiring Patches:
The patch involves changes to several files within the `aten/src/ATen/native/quantized/` directory. Here's a list of the specific files and the lines where the modifications are marked by the comment "Yeseong":

- `./aten/src/ATen/native/quantized/cpu/quantized_ops.h:166`
- `./aten/src/ATen/native/quantized/cpu/quantized_ops.h:205`
- `./aten/src/ATen/native/quantized/cpu/qthreshold.cpp:13`
- `./aten/src/ATen/native/quantized/cpu/qthreshold.cpp:38`
- `./aten/src/ATen/native/quantized/cpu/qthreshold.cpp:59`
- `./aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:3604`
- `./aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:3792`
- `./aten/src/ATen/native/quantized/library.cpp:60`
