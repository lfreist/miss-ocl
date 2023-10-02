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

namespace mcl {

// ===== Kernel ========================================================================================================
Kernel::Kernel(mcl::Environment& environment, cl::NDRange range, std::string name, const std::string& cl_c_source)
    : _name(std::move(name)), _environment(&environment) {
  set_range(range);
  const std::string source_code(_device_capabilities + cl_c_source);
  cl::Program::Sources sources;
  sources.push_back(source_code);
  cl::Program cl_program(_environment->_cl_context, sources);
  std::string build_options("-cl-fast-relaxed-math");
  // std::string build_options("-cl-std=CL1.2");
  if (_environment->_device->intel_gt_4gb_buffer_required()) {
    build_options.append(" -cl-intel-greater-than-4GB-buffer-required");
  }
  int error = cl_program.build(_environment->_device->get_cl_device(), build_options.c_str());
  check_opencl_error(error);
  _cl_kernel = cl::Kernel(cl_program, _name.c_str(), &error);
  check_opencl_error(error);
}

void Kernel::set_range(uint64_t x, uint64_t y, uint64_t z) {
  _cl_global_range = cl::NDRange(x, y, z);
  _cl_local_range = cl::NDRange(WORKGROUP_SIZE);
}

void Kernel::set_range(cl::NDRange global, cl::NDRange local) {
  _cl_global_range = global;
  _cl_local_range = local;
}


void Kernel::enqueue_run(unsigned int t, const std::vector<cl::Event>* event_waitlist, cl::Event* event_returned) {
  for (unsigned i = 0; i < t; ++i) {
    int error = _environment->_cl_queue.enqueueNDRangeKernel(_cl_kernel, cl::NullRange, _cl_global_range,
                                                             _cl_local_range, event_waitlist, event_returned);
    check_opencl_error(error);
  }
}

void Kernel::run(unsigned int t, const std::vector<cl::Event>* event_waitlist, cl::Event* event_returned) {
  enqueue_run(t, event_waitlist, event_returned);
  finish_queue();
}

void Kernel::finish_queue() { _environment->_cl_queue.finish(); }

}  // namespace mcl