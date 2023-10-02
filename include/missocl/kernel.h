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

#include <missocl/utils.h>

#include <CL/opencl.hpp>
#include <cstdint>
#include <string>

namespace mcl {
class Environment;
template <unsigned dimension, typename T>
class Memory;

class Kernel {
  friend class Environment;

 public:
  void set_range(uint64_t x, uint64_t y = 0, uint64_t z = 0);
  void set_range(cl::NDRange global, cl::NDRange local = cl::NDRange(64));

  template <typename... T>
  void set_parameters(const T&... parameters) {
    link_parameters(parameters...);
  }

  template <typename... T>
  void set_args(const T&... args) {
    link_args(args...);
  }

  void enqueue_run(unsigned t = 1, const std::vector<cl::Event>* event_waitlist = nullptr,
                   cl::Event* returned_event = nullptr);

  void run(unsigned t = 1, const std::vector<cl::Event>* event_waitlist = nullptr, cl::Event* event_returned = nullptr);

  void finish_queue();

 private:
  Kernel(Environment& environment, cl::NDRange range, std::string name, const std::string& cl_c_source = "");

  template <typename T0, typename... Tn>
  void link_args(const T0& arg, const Tn&... args) {
    link_arg(arg);
    link_args(args...);
  }

  template <typename T>
  void link_arg(const T& arg) {
    int error = _cl_kernel.setArg(_parameter_count++, arg);
    check_opencl_error(error);
  }

  void link_args() {}

  template <unsigned dimension, typename T>
  void link_parameter(const Memory<dimension, T>& memory) {
    int error = _cl_kernel.setArg(_parameter_count++, memory.get_cl_buffer());
    check_opencl_error(error);
  }

  template <typename T>
  void link_parameter(const T& parameter) {
    int error = _cl_kernel.setArg(_parameter_count++, sizeof(T), static_cast<void*>(parameter));
    check_opencl_error(error);
  }

  void link_parameters() {}  // catch last recursive call of link_parameters(...)

  template <typename T0, typename... Tn>
  void link_parameters(const T0& parameter, const Tn&... parameters) {
    link_parameter(parameter);
    link_parameters(parameters...);
  }

  std::string _name;
  cl::Kernel _cl_kernel;
  Environment* _environment;
  cl::NDRange _cl_global_range;
  cl::NDRange _cl_local_range;
  cl_uint _parameter_count{0};

  const std::string _device_capabilities{
      "#define def_workgroup_size 64\n"
      "#ifdef cl_khr_fp64\n"
      "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
      "#endif\n"
      "#ifdef cl_khr_fp16\n"
      "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
      "#endif\n"
      "#ifdef cl_khr_int64_base_atomics\n"
      "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
      "#endif\n\n"};
};

}  // namespace mcl