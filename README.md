# miss-ocl
miss-ocl (Make It Simple and Stupid OpenCL) is an [OpenCL](https://www.khronos.org/opencl/) wrapper for C++ that aims to
make OpenCL more intuitive and simple to use.

## What does miss-ocl provide?
1. A global `DeviceManager` instance that allows to retrieve OpenCL devices in just one line
2. A ready to use `Environment` object that brings `Device`, `cl::Context`, `cl::Program` and `cl::CommandQueue`
   together and provides a simple `Environment::add_kernel(...)` function for creating a kernel
3. A `Memory` object that allows
   - memory usage tracking per `Device`
   - 1, 2 and 3 dimensional implementations for simpler usage
4. The `KERNEL_CODE(name, ...)` Macro that allows to write inline Kernel code.

## How is miss-ocl structured?
The core of miss-ocl is the `Environment` that manages everything you need to do by hand when using OpenCL:

An `Environment` holds a `Device` bond to a `cl::Context` and the corresponding `cl::CommandQueue`
A Kernel can only be created via an `Environment` instance using the `Environment::add_kernel(...)` function.
