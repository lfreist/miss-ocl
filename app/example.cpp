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

#include <chrono>


KERNEL_CODE(
    matrix_multiplication,
    __kernel void matrix_multiplication(__global const float* A, __global const float* B, __global float* C,
                                        const int A_n, const int A_m, const int B_n, const int B_m) {
      int row = get_global_id(0);
      int col = get_global_id(1);
      float sum = 0.0f;
      for (int k = 0; k < A_m; ++k) {
        sum += A[row * A_m + k] * B[k * B_m + col];
      }
      C[row * B_m + col] = sum;
    });

KERNEL_CODE(
    vector_add, __kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {
      int i = get_global_id(0);
      C[i] = A[i] + B[i];
    });

int main(int argc, char** argv) {
  size_t size = 1024;

  mcl::Environment env;

  mcl::Memory<2, float> A(&env, size, size, 3);
  mcl::Memory<2, float> B(&env, size, size, 4);
  mcl::Memory<2, float> C(&env, size, size);

  // auto _kernel = env.add_kernel(size, "matrix_multiplication", matrix_multiplication);
  auto _kernel = env.add_kernel(cl::NDRange(size, size), "matrix_multiplication", std::filesystem::path("../../app/matrix_mul.cl"));
  _kernel.set_parameters(A, B, C);          // pointer arguments
  _kernel.set_args(size, size, size, size); // integer arguments

  A.write_to_device();
  B.write_to_device();

  // std::cout << C.str() << std::endl;
  auto computing_start = std::chrono::high_resolution_clock::now();

  _kernel.run();

  auto computing_stop = std::chrono::high_resolution_clock::now();

  C.read_from_device();

  // std::cout << C.str() << std::endl;
  std::cout << C[0] << std::endl;

  std::cout << "Computing Time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(computing_stop - computing_start) << std::endl;
  return 0;
}