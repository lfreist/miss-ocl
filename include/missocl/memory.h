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

#include <missocl/environment.h>
#include <missocl/utils.h>

#include <concepts>
#include <cstddef>
#include <ostream>
#include <type_traits>
#include <vector>

namespace mcl {

template <unsigned dimensions, typename T>
class Memory {};

template <typename T>
class Memory<1, T> {
 public:
  struct Range {
    explicit Range(size_t x_size_) : x_size(x_size_) {}
    size_t x_size;
    size_t y_size{0};
    size_t z_size{0};
  };

  Memory(Environment* environment, T* data, size_t size)
      : _environment(environment), _data(data), _range(size), _unowned_data(true) {
    _allocate_device_buffer();
  }

  Memory(Environment* environment, size_t x_size, T default_value = static_cast<T>(0))
      : _environment(environment), _range(x_size) {
    _data = new T[size()];
    std::fill_n(_data, size(), default_value);
    _allocate_device_buffer();
  }

  ~Memory() {
    if (!_unowned_data) {
      delete[] _data;
    }
  }

  T* data() { return _data; }
  const T* data() const { return _data; }
  [[nodiscard]] size_t size() const { return _range.x_size; }
  [[nodiscard]] constexpr unsigned dimension() const { return 1; };
  [[nodiscard]] size_t mem_size() const { return size() * sizeof(T); }
  T& operator[](size_t i) { return _data[i]; }
  const T& operator[](size_t i) const { return _data[i]; }
  T& at(size_t x) { return _data[x]; }
  const T& at(size_t x) const { return _data[x]; }
  void assign(T* data, size_t size) {
    if (!_unowned_data) {
      delete[] _data;
    }
    _unowned_data = true;
    _data = data;
    _range.x_size = size;
    _allocate_device_buffer();
  }

  [[nodiscard]] const cl::Buffer& get_cl_buffer() const { return _device_buffer; }

  void reset(T default_value = static_cast<T>(0)) {
    std::fill_n(_data, size(), default_value);
    write_to_device();
  }

  void write_to_device(bool blocking = true, const std::vector<cl::Event>* event_waitlist = nullptr,
                       cl::Event* event_returned = nullptr) {
    _environment->_cl_queue.enqueueWriteBuffer(_device_buffer, blocking, 0, mem_size(), _data, event_waitlist,
                                               event_returned);
  }

  void read_from_device(bool blocking = true, const std::vector<cl::Event>* event_waitlist = nullptr,
                        cl::Event* event_returned = nullptr) {
    _environment->_cl_queue.enqueueReadBuffer(_device_buffer, blocking, 0, mem_size(), static_cast<void*>(_data),
                                              event_waitlist, event_returned);
  }

 private:
  void _allocate_device_buffer() {
    _environment->_device->_memory_used_Bytes += mem_size();
    int error = 0;
    if (_environment->_device->intel_gt_4gb_buffer_required()) {
      // https://github.com/intel/compute-runtime/blob/master/programmers-guide/ALLOCATIONS_GREATER_THAN_4GB.md
      _device_buffer =
          cl::Buffer(_environment->_cl_context, CL_MEM_READ_WRITE | ((int)1 << 23), mem_size(), nullptr, &error);
    } else {
      _device_buffer = cl::Buffer(_environment->_cl_context, CL_MEM_READ_WRITE, mem_size(), nullptr, &error);
    }
    check_opencl_error(error);
    _device_buffer_init = true;
  }

  Environment* _environment;
  T* _data;
  bool _unowned_data{false};
  Range _range;
  cl::Buffer _device_buffer;
  bool _device_buffer_init{false};
};

template <typename T>
class Memory<2, T> {
 public:
  struct Range {
    explicit Range(size_t x_size_, size_t y_size_) : x_size(x_size_), y_size(y_size_) {}
    size_t x_size;
    size_t y_size;
    size_t z_size{0};
  };

  Memory(Environment* environment, T* data, size_t x_size, size_t y_size)
      : _environment(environment), _data(data), _range(x_size, y_size), _unowned_data(true) {
    _allocate_device_buffer();
  }

  Memory(Environment* environment, size_t x_size, size_t y_size, T default_value = static_cast<T>(0))
      : _environment(environment), _range(x_size, y_size) {
    _data = new T[size()];
    std::fill_n(_data, size(), default_value);
    _allocate_device_buffer();
  }

  ~Memory() {
    if (!_unowned_data) {
      delete[] _data;
    }
  }

  T* data() { return _data; }
  const T* data() const { return _data; }
  [[nodiscard]] size_t size() const { return _range.x_size * _range.y_size; }
  [[nodiscard]] constexpr unsigned dimension() const { return 2; };
  [[nodiscard]] size_t mem_size() const { return size() * sizeof(T); }
  T& operator[](size_t i) { return _data[i]; }
  const T& operator[](size_t i) const { return _data[i]; }
  T& at(size_t x, size_t y) { return _data[_range.x_size * y + x]; }
  const T& at(size_t x, size_t y) const { return at(x, y); }
  void assign(T* data, size_t x_size, size_t y_size) {
    if (!_unowned_data) {
      delete[] _data;
    }
    _unowned_data = true;
    _data = data;
    _range.x_size = x_size;
    _range.y_size = y_size;
    _allocate_device_buffer();
  }

  [[nodiscard]] const cl::Buffer& get_cl_buffer() const { return _device_buffer; }

  void reset(T default_value = static_cast<T>(0)) {
    std::fill_n(_data, size(), default_value);
    write_to_device();
  }

  void write_to_device(bool blocking = true, const std::vector<cl::Event>* event_waitlist = nullptr,
                       cl::Event* event_returned = nullptr) {
    _environment->_cl_queue.enqueueWriteBuffer(_device_buffer, blocking, 0, mem_size(), _data, event_waitlist,
                                               event_returned);
  }

  void read_from_device(bool blocking = true, const std::vector<cl::Event>* event_waitlist = nullptr,
                        cl::Event* event_returned = nullptr) {
    _environment->_cl_queue.enqueueReadBuffer(_device_buffer, blocking, 0, mem_size(), static_cast<void*>(_data),
                                              event_waitlist, event_returned);
  }

  [[nodiscard]] std::string str() const {
    std::stringstream ss;
    for (size_t i = 0; i < size() - 1; ++i) {
      ss << this->operator[](i) << ' ';
    }
    ss << this->operator[](size() - 1);
    return ss.str();
  }

 private:
  [[maybe_unused]] void _allocate_device_buffer() {
    _environment->_device->_memory_used_Bytes += mem_size();
    int error = 0;
    if (_environment->_device->intel_gt_4gb_buffer_required()) {
      // https://github.com/intel/compute-runtime/blob/master/programmers-guide/ALLOCATIONS_GREATER_THAN_4GB.md
      _device_buffer =
          cl::Buffer(_environment->_cl_context, CL_MEM_READ_WRITE | ((int)1 << 23), mem_size(), nullptr, &error);
    } else {
      _device_buffer = cl::Buffer(_environment->_cl_context, CL_MEM_READ_WRITE, mem_size(), nullptr, &error);
    }
    check_opencl_error(error);
    _device_buffer_init = true;
  }

  T* _data;
  bool _unowned_data{false};
  Range _range;
  cl::Buffer _device_buffer;
  bool _device_buffer_init{false};
  Environment* _environment;
};

template <typename T>
class Memory<3, T> {
 public:
  struct Range {
    explicit Range(size_t x_size_, size_t y_size_, size_t z_size_)
        : x_size(x_size_), y_size(y_size_), z_size(z_size_) {}
    size_t x_size;
    size_t y_size;
    size_t z_size;
  };
  Memory(Environment* environment, T* data, size_t x_size, size_t y_size, size_t z_size)
      : _environment(environment), _data(data), _range(x_size, y_size, z_size), _unowned_data(true) {}

  Memory(Environment* environment, size_t x_size, size_t y_size, size_t z_size, T default_value = static_cast<T>(0))
      : _environment(environment), _range(x_size, y_size, z_size) {
    _data = new T[size()];
    std::fill_n(_data, size(), default_value);
    _allocate_device_buffer();
  }

  ~Memory() {
    if (!_unowned_data) {
      delete[] _data;
    }
  }

  T* data() { return _data; }
  const T* data() const { return _data; }
  [[nodiscard]] size_t size() const { return _range.x_size * _range.y_size * _range.z_size; }
  [[nodiscard]] constexpr unsigned dimension() const { return 3; };
  [[nodiscard]] size_t mem_size() const { return size() * sizeof(T); }
  T& operator[](size_t i) { return _data[i]; }
  const T& operator[](size_t i) const { return _data[i]; }
  T& at(size_t x, size_t y, size_t z) { return _data[_range.x_size * y + _range.y_size * z + x]; }
  const T& at(size_t x, size_t y) const { return at(x, y); }
  void assign(T* data, size_t x_size, size_t y_size, size_t z_size) {
    if (!_unowned_data) {
      delete[] _data;
    }
    _unowned_data = true;
    _data = data;
    _range.x_size = x_size;
    _range.y_size = y_size;
    _range.z_size = z_size;
    _allocate_device_buffer();
  }

  [[nodiscard]] const cl::Buffer& get_cl_buffer() const { return _device_buffer; }

  void reset(T default_value = static_cast<T>(0)) {
    std::fill_n(_data, size(), default_value);
    write_to_device();
  }

  void write_to_device(bool blocking = true, const std::vector<cl::Event>* event_waitlist = nullptr,
                       cl::Event* event_returned = nullptr) {
    _environment->_cl_queue.enqueueWriteBuffer(_device_buffer, blocking, 0, mem_size(), _data, event_waitlist,
                                               event_returned);
  }

  void read_from_device(bool blocking = true, const std::vector<cl::Event>* event_waitlist = nullptr,
                        cl::Event* event_returned = nullptr) {
    _environment->_cl_queue.enqueueReadBuffer(_device_buffer, blocking, 0, mem_size(), static_cast<void*>(_data),
                                              event_waitlist, event_returned);
  }

 private:
  [[maybe_unused]] [[maybe_unused]] void _allocate_device_buffer() {
    _environment->_device->_memory_used_Bytes += mem_size();
    int error = 0;
    if (_environment->_device->intel_gt_4gb_buffer_required()) {
      // https://github.com/intel/compute-runtime/blob/master/programmers-guide/ALLOCATIONS_GREATER_THAN_4GB.md
      _device_buffer =
          cl::Buffer(_environment->_cl_context, CL_MEM_READ_WRITE | ((int)1 << 23), mem_size(), nullptr, &error);
    } else {
      _device_buffer = cl::Buffer(_environment->_cl_context, CL_MEM_READ_WRITE, mem_size(), nullptr, &error);
    }
    check_opencl_error(error);
    _device_buffer_init = true;
  }

  T* _data;
  bool _unowned_data{false};
  Range _range;
  cl::Buffer _device_buffer;
  bool _device_buffer_init{false};
  Environment* _environment;
};

}  // namespace mcl