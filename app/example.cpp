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

const std::string kernel_code(
    "__kernel void add_kernel(global float* A, global float* B, global float* C) {"
    "const uint n = get_global_id(0);"
    "C[n] = A[n]+B[n];"
    "}");

int main(int argc, char** argv) {
  size_t size = 1024 * 1024 * 1024;

  mcl::Environment env;

  mcl::Memory<float> A(env, size, 1, true, 3.4f);
  mcl::Memory<float> B(env, size, 1, true, 2.5f);
  mcl::Memory<float> C(env, size);

  auto kernel = env.add_kernel(size, "add_kernel", kernel_code);
  kernel.set_parameters(A, B, C);

  A.write_to_device();
  B.write_to_device();

  std::cout << "C[0] " << C[0] << std::endl;

  auto computing_start = std::chrono::high_resolution_clock::now();

  kernel.run();

  auto computing_stop = std::chrono::high_resolution_clock::now();

  C.read_from_device();

  std::cout << "C[0] " << C[0] << std::endl;

  std::cout << "Computing Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(computing_stop - computing_start) << std::endl;
}