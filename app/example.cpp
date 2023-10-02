/**
 * Author: Leon Freist <freist.leon@gmail.com>
 *
 * This file belongs to MISS-OCL.
 * MISS-OCL is licensed under
 *  A) MIT-License:    Copyright (c) 2023 Leon Freist
 *  B) The Unlicensed: This is free and unencumbered software released into the
 *     public domain
 * You are free to chose the license whichever you prefer.
 */

#include <missocl/opencl.h>
#include <missocl/utils.h>

#include <chrono>

void matrix_multiplication() {
  KERNEL_CODE(
      mmul, __kernel void mmul(__global const float* A, __global const float* B, __global float* C, const int A_n,
                               const int A_m, const int B_n, const int B_m) {
        if (A_n != B_m) {
          return;
        }
        int row = get_global_id(0);
        int col = get_global_id(1);
        float sum = 0.0f;
        for (int k = 0; k < A_m; ++k) {
          sum += A[row * A_m + k] * B[k * B_m + col];
        }
        C[row * B_m + col] = sum;
      });
  mcl::Timer timer;
  const int size = 2048;
  mcl::Environment env;
  mcl::Memory<2, float> A(&env, size, size, 3);
  mcl::Memory<2, float> B(&env, size, size, 4);
  mcl::Memory<2, float> C(&env, size, size);
  std::cout << "--- Matrix Multiplication ---\n";
  std::cout << "Matrix size: " << size << "x" << size << " (" + std::to_string(A.mem_size() / 1024 / 1024) + " MiB)"
            << std::endl;
  std::cout << "Device:      " << *env.get_device() << std::endl;
  std::cout << "-----------------------------\n";
  auto kernel = env.add_kernel(cl::NDRange(size, size), "mmul", mmul);
  kernel.set_parameters(A, B, C);           // pointer arguments
  kernel.set_args(size, size, size, size);  // integer arguments
  A.write_to_device();
  B.write_to_device();
  std::cout << "  Computing Matrix Product:" << std::endl;
  std::cout << "                    C[0][0] ... C[512][0] ... C[1023][0]" << std::endl;
  std::cout << "    initial values:    " << C[0] << "    ...     " << C[512] << "     ...     " << C[1023] << std::endl;
  timer.start();
  kernel.run();
  auto dur = timer.stop();
  C.read_from_device();
  std::cout << "    result:          " << C[0] << "  ...   " << C[512] << "   ...    " << C[1023] << std::endl;
  std::cout << "  Computation time: " << dur << std::endl;
}

void vector_addition() {
  KERNEL_CODE(
      vadd, __kernel void vadd(__global const float* A, __global const float* B, __global float* C) {
        int i = get_global_id(0);
        C[i] = A[i] + B[i];
      });
  mcl::Timer timer;
  size_t size = 1024 << 8;
  mcl::Environment env;
  mcl::Memory<1, float> A(&env, size, 3);
  mcl::Memory<1, float> B(&env, size, 4);
  mcl::Memory<1, float> C(&env, size);
  std::cout << "--- Vector Addition ---\n";
  std::cout << "Vector size: " << size << " (" + std::to_string(A.mem_size() / 1024 / 1024) + " MiB)" << std::endl;
  std::cout << "Device:      " << *env.get_device() << std::endl;
  std::cout << "-----------------------------\n";
  auto _kernel = env.add_kernel(cl::NDRange(size), "vadd", vadd);
  _kernel.set_parameters(A, B, C);  // pointer arguments
  A.write_to_device();
  B.write_to_device();
  std::cout << "  Computing Vector Addition:" << std::endl;
  std::cout << "                    C[0] ... C[512] ... C[1023]" << std::endl;
  std::cout << "    initial values:  " << C[0] << "   ...  " << C[512] << "     ...  " << C[1023] << std::endl;
  timer.start();
  _kernel.run();
  auto dur = timer.stop();
  C.read_from_device();
  std::cout << "    result:          " << C[0] << "   ...  " << C[512] << "    ...   " << C[1023] << std::endl;
  std::cout << "  Computation time: " << dur << std::endl;
}

void print_help() {
  std::cout << "Usage: ./example [ARG]\n"
            << "\n"
            << "  vadd   run vector addition\n"
            << "  mmul   run matrix multiplication\n"
            << "\n"
            << "  If nor argument is provided vadd and mmul are run.\n";
}

int main(int argc, char** argv) {
  if (argc == 1) {
    matrix_multiplication();
    vector_addition();
  } else if (argc == 2) {
    std::string arg(argv[1]);
    if (arg == "mmul") {
      matrix_multiplication();
    } else if (arg == "vadd") {
      vector_addition();
    } else {
      print_help();
    }
  } else {
    print_help();
  }
  return 0;
}