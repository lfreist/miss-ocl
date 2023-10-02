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

#include <missocl/device.h>
#include <missocl/environment.h>
#include <missocl/kernel.h>
#include <missocl/utils.h>

#include <fstream>

namespace mcl {

// ===== Environment ===================================================================================================
Environment::Environment() : _device(DeviceManager::get<Filter::MAX_FLOPS>()) { _init(); }

Environment::Environment(Device& device) : _device(&device) { _init(); }

Environment::Environment(Device* device) : _device(device) { _init(); }

Kernel Environment::add_kernel(cl::NDRange range, std::string name, const std::string& cl_c_source) {
  return {*this, range, std::move(name), cl_c_source};
}

Kernel Environment::add_kernel(cl::NDRange range, std::string name, const std::filesystem::path& cl_c_source_file) {
  std::ifstream file(cl_c_source_file);
  if (!file) {
    std::cerr << "Could not read file '" << cl_c_source_file << "'." << std::endl;
  }
  return {*this, range, std::move(name), {std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>())}};
}

const Device* Environment::get_device() const { return _device; }

void Environment::_init() {
  cl_int error;
  _cl_context = cl::Context(_device->get_cl_device(), nullptr, nullptr, nullptr, &error);
  check_opencl_error(error);
  _cl_queue = cl::CommandQueue(_cl_context, _device->get_cl_device(), 0, &error);
  check_opencl_error(error);
}

}  // namespace mcl