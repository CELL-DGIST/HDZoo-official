# PyTorch patch to support the sign function for quantized data type
This patch is made on torch 1.20.0. It includes the entire directory of `aten/src/ATen/native/quantized/` in the PyTorch code.
So, related license or all other rights are reserved as in the [original PyTorch repo](https://github.com/pytorch/pytorch).


## How to patch?
- If you are using the same version, you can overwrite this directory in the ./aten/src/ATen/native/quantized/ of your pytorch source code.
- Otherwise, to get safe build, find the comments marked with "Yeseong", and apply them on your pytorch Source code.
(USE: # grep -nr 'Yeseong' .)

Here's the list of files that need the patches.
```
./aten/src/ATen/native/quantized/cpu/quantized_ops.h:166:// Yeseong
./aten/src/ATen/native/quantized/cpu/quantized_ops.h:205:DECLARE_DISPATCH(qsign_fn, qsign_stub); // Yeseong
./aten/src/ATen/native/quantized/cpu/qthreshold.cpp:13:DEFINE_DISPATCH(qsign_stub); // Yeseong
./aten/src/ATen/native/quantized/cpu/qthreshold.cpp:38:// Yeseong
./aten/src/ATen/native/quantized/cpu/qthreshold.cpp:59:  m.impl(TORCH_SELECTIVE_NAME("quantized::sign"), TORCH_FN(sign_quantized_cpu)); // Yeseong
./aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:3604:// Yeseong
./aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:3792:// Yeseong
Binary file ./aten/src/ATen/native/quantized/cpu/kernels/.QuantizedOpKernels.cpp.swp matches
./aten/src/ATen/native/quantized/library.cpp:60:  m.def(TORCH_SELECTIVE_SCHEMA("quantized::sign(Tensor qx) -> Tensor qy")); //
```
