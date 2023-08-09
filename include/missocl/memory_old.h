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

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>

namespace mcl {

template <typename T>
class Memory {
 public:
  struct Buffer {
    T* data{nullptr};
    size_t size{0};
    bool initialized{false};
    bool unowned{false};

    explicit Buffer(size_t size_) {
      size = size_;
      data = new T[size];
      initialized = true;
    }

    Buffer(size_t size_, T default_value) {
      size = size_;
      data = new T[size];
      std::fill_n(data, size, default_value);
      initialized = true;
    }

    Buffer(T* data_, size_t size_) {
      data = data_;
      size = size_;
      initialized = true;
      unowned = true;
    }

    Buffer(Buffer&& buffer) noexcept : size(buffer.size), initialized(buffer.initialized), data(buffer.data) {
      buffer.size = 0;
      buffer.initialized = false;
      buffer.data = nullptr;
    }

    Buffer& operator=(Buffer&& buffer) noexcept {
      if (this != &buffer) {
        delete[] data;
        data = buffer.data;
        size = buffer.size;
        initialized = buffer.initialized;
        buffer.data = nullptr;
        buffer.size = 0;
        buffer.initialized = false;
      }
      return *this;
    }

    Buffer(const Buffer& buffer) : size(buffer.size), initialized(buffer.initialized), data(new T[buffer.size]) {
      std::memcpy(data, buffer.data, size);
    }

    Buffer& operator=(const Buffer& buffer) {
      if (this != &buffer) {
        if (size != buffer.size) {
          delete[] data;
          size = buffer.size;
          data = new T[size];
        }
        initialized = buffer.initialized;
        std::memcpy(data, buffer.data, size);
      }
      return *this;
    }

    ~Buffer() { delete[] data; }

    void destruct() {
      if (!unowned) {
        delete[] data;
      }
    }

    void set_data(T* data_, size_t size_) {
      destruct();
      data = data_;
      size = size_;
      initialized = true;
      unowned = true;
    }
  };
  /**
   * @brief
   * @param device
   * @param cl_context
   * @param size
   * @param dimensions
   * @param allocate_device
   * @param default_value
   */
  Memory(Environment& environment, size_t size, uint32_t dimensions = 1, bool allocate_device = true,
         T default_value = static_cast<T>(0))
      : _size(size),
        _environment(&environment),
        _dimensions(dimensions == 0 ? 1 : dimensions /*dimension cannot be 0*/),
        _host_buffer(size * dimensions, default_value) {
    if (allocate_device) {
      _allocate_device_buffer();
    }
    _init_auxiliary_pointers();
    write_to_device();
  }

  [[nodiscard]] const cl::Buffer& get_cl_buffer() const { return _device_buffer; }

  Buffer swap_host_buffer(Buffer buffer) {
    Buffer swap = std::move(_host_buffer);
    _host_buffer = std::move(buffer);
    return swap;
  }

  void add_missing_buffer() {
    if (_host_buffer.initialized) {
      if (!_device_buffer_exists) {
        // add device buffer
        _allocate_device_buffer();
        write_to_device();
      }
    } else {
      if (_device_buffer_exists) {
        if (!_host_buffer.initialized) {
          // add host buffer
          _host_buffer = Buffer(range(), static_cast<T>(0));
          read_from_device();
        }
      }
    }
  }

  void delete_host_buffer() {
    _host_buffer.destruct();
    if (!_device_buffer_exists) {
      _size = 0;
      _dimensions = 1;
    }
  }

  void delete_device_buffer() {
    if (_device_buffer_exists) {
      _environment->_device->_memory_used_Bytes -= mem_size();
    }
    _device_buffer_exists = false;
    _device_buffer = nullptr;
    if (!_host_buffer.initialized) {
      _size = 0;
      _dimensions = 1;
    }
  }

  void delete_buffers() {
    delete_device_buffer();
    delete_host_buffer();
  }

  void reset(T default_value = static_cast<T>(0)) {
    if (_host_buffer.initialized) {
      std::fill_n(_host_buffer.data, _host_buffer.size, default_value);
    }
    write_to_device();
  }

  void read_from_device(bool blocking = true, const std::vector<cl::Event>* event_waitlist = nullptr,
                        cl::Event* event_returned = nullptr) {
    if (_host_buffer.initialized && _device_buffer_exists) {
      _environment->_cl_queue.enqueueReadBuffer(_device_buffer, blocking, 0, mem_size(),
                                                static_cast<void*>(_host_buffer.data), event_waitlist, event_returned);
    }
  }

  void read_from_device(size_t offset, size_t size, bool blocking = true,
                        const std::vector<cl::Event>* event_waitlist = nullptr, cl::Event* event_returned = nullptr) {
    if (_host_buffer.initialized && _device_buffer_exists) {
      size_t safe_offset = std::min(offset, range());
      int64_t safe_size = std::min(size, range() - safe_offset);
      if (safe_size > 0) {
        _environment->_cl_queue.enqueueReadBuffer(
            _device_buffer, blocking, safe_offset * sizeof(T), safe_size * sizeof(T),
            static_cast<void*>(_host_buffer.data + safe_offset), event_waitlist, event_returned);
      }
    }
  }

  void write_to_device(bool blocking = true, const std::vector<cl::Event>* event_waitlist = nullptr,
                       cl::Event* event_returned = nullptr) {
    if (_host_buffer.initialized && _device_buffer_exists) {
      _environment->_cl_queue.enqueueWriteBuffer(_device_buffer, blocking, 0, mem_size(), _host_buffer.data,
                                                 event_waitlist, event_returned);
    }
  }

  void write_to_device(size_t offset, size_t size, bool blocking = true,
                       const std::vector<cl::Event>* event_waitlist = nullptr, cl::Event* event_returned = nullptr) {
    if (_host_buffer.initialized && _device_buffer_exists) {
      size_t safe_offset = std::min(offset, range());
      int64_t safe_size = std::min(size, range() - safe_offset);
      if (safe_size > 0) {
        _environment->_cl_queue.enqueueWriteBuffer(
            _device_buffer, blocking, safe_offset * sizeof(T), safe_size * sizeof(T),
            static_cast<void*>(_host_buffer.data + safe_offset), event_waitlist, event_returned);
      }
    }
  }

  [[nodiscard]] size_t size() const { return _size; }
  [[nodiscard]] uint32_t dimensions() const { return _dimensions; }
  [[nodiscard]] uint32_t range() const { return _size * _dimensions; }
  [[nodiscard]] size_t mem_size() const { return _size * _dimensions * sizeof(T); }
  T* data() { return _host_buffer.data; }
  const T* data() const { return _host_buffer.data; }
  Buffer& host_buffer() { return _host_buffer; }
  const Buffer& host_buffer() const { return _host_buffer; }
  T& operator[](size_t i) {
    if (i >= _host_buffer.size) {
      throw std::out_of_range(std::to_string(i) + " but size is " + std::to_string(_host_buffer.size));
    }
    return _host_buffer.data[i];
  }
  const T& operator[](size_t i) const {
    if (i >= _host_buffer.size) {
      throw std::out_of_range(std::to_string(i) + " but size is " + std::to_string(_host_buffer.size));
    }
    return _host_buffer.data[i];
  }

 private:
  void _init_auxiliary_pointers() {
    _x = _s0 = _host_buffer.data;
    if (_dimensions > 0x1) _y = _s1 = _host_buffer.data + _size * 0x1;
    if (_dimensions > 0x2) _z = _s2 = _host_buffer.data + _size * 0x2;
    if (_dimensions > 0x3) _w = _s3 = _host_buffer.data + _size * 0x3;
    if (_dimensions > 0x4) _s4 = _host_buffer.data + _size * 0x4;
    if (_dimensions > 0x5) _s5 = _host_buffer.data + _size * 0x5;
    if (_dimensions > 0x6) _s6 = _host_buffer.data + _size * 0x6;
    if (_dimensions > 0x7) _s7 = _host_buffer.data + _size * 0x7;
    if (_dimensions > 0x8) _s8 = _host_buffer.data + _size * 0x8;
    if (_dimensions > 0x9) _s9 = _host_buffer.data + _size * 0x9;
    if (_dimensions > 0xA) _sA = _host_buffer.data + _size * 0xA;
    if (_dimensions > 0xB) _sB = _host_buffer.data + _size * 0xB;
    if (_dimensions > 0xC) _sC = _host_buffer.data + _size * 0xC;
    if (_dimensions > 0xD) _sD = _host_buffer.data + _size * 0xD;
    if (_dimensions > 0xE) _sE = _host_buffer.data + _size * 0xE;
    if (_dimensions > 0xF) _sF = _host_buffer.data + _size * 0xF;
  }

  [[maybe_unused]] void _allocate_device_buffer() {
    _environment->_device->_memory_used_Bytes += mem_size();
    if (_environment->_device->memory_used_Bytes() > _environment->_device->memory_Bytes()) {
      std::cerr << "Warning: There may be not enough memory available for allocations of this size." << std::endl;
    }
    int error = 0;
    if (_environment->_device->intel_gt_4gb_buffer_required()) {
      // https://github.com/intel/compute-runtime/blob/master/programmers-guide/ALLOCATIONS_GREATER_THAN_4GB.md
      _device_buffer =
          cl::Buffer(_environment->_cl_context, CL_MEM_READ_WRITE | ((int)1 << 23), mem_size(), nullptr, &error);
    } else {
      _device_buffer = cl::Buffer(_environment->_cl_context, CL_MEM_READ_WRITE, mem_size(), nullptr, &error);
    }
#ifdef __MCL_DEBUG__
    if (error) {
      std::cerr << "Error allocating device memory: " << cl_error(error) << std::endl;
    }
#endif
    _device_buffer_exists = true;
  }

  size_t _size;
  uint32_t _dimensions;
  bool _device_buffer_exists{false};
  Buffer _host_buffer;
  cl::Buffer _device_buffer;
  Environment* _environment;

  T* _x{nullptr};
  T* _y{nullptr};
  T* _z{nullptr};
  T* _w{nullptr};

  T* _s0{nullptr};
  T* _s1{nullptr};
  T* _s2{nullptr};
  T* _s3{nullptr};
  T* _s4{nullptr};
  T* _s5{nullptr};
  T* _s6{nullptr};
  T* _s7{nullptr};
  T* _s8{nullptr};
  T* _s9{nullptr};
  T* _sA{nullptr};
  T* _sB{nullptr};
  T* _sC{nullptr};
  T* _sD{nullptr};
  T* _sE{nullptr};
  T* _sF{nullptr};
};

}  // namespace mcl