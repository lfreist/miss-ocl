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

#pragma once

#include <CL/opencl.hpp>
#include <cstdint>
#include <string>
#include <filesystem>

namespace mcl {
class Kernel;
class Device;

// ===== Environment ===================================================================================================
class Environment {
  template <unsigned dimension, typename T>
  friend class Memory;
  friend class Kernel;

 public:
  Environment();
  explicit Environment(Device& device);
  explicit Environment(Device* device);

  Kernel add_kernel(cl::NDRange range, std::string name, const std::string& cl_c_source);
  Kernel add_kernel(cl::NDRange range, std::string name, const std::filesystem::path& cl_c_source_file);

  [[nodiscard]] const Device* get_device() const;

 private:
  void _init();

  cl::Context _cl_context{};
  Device* _device;
  cl::CommandQueue _cl_queue{};
};

}  // namespace mcl