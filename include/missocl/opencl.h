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
#include <algorithm>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <missocl/utils.h>

#define __MCL_VERSION__ "1.0.0"
#define __MCL_DEBUG__

namespace mcl {

#define WORKGROUP_SIZE 64

/**
 * @brief Device represents a single OpenCL Device.
 *        It provides instant methods for retrieving common data.
 */
class Device {
  template <typename T>
  friend class Memory;
  friend class DeviceManager;
  friend std::ostream& operator<<(std::ostream& os, const Device& device);

 public:
  /**
   * @brief Device can only be constructed using an explicit id, a cl::Device and a cl::Context.
   *
   *        It is highly recommended, not to construct devices by hand but rather by using the DeviceManager class.
   */
  Device(uint32_t id, cl::Device cl_device);
  ~Device();

  /// Copy Constructor
  Device(const Device& device) = delete;
  /// Copy Assignment Operator
  Device& operator=(const Device& device) = delete;

  /// Move Constructor
  Device(Device&& device) noexcept;
  /// Move Assignment Operator
  Device& operator=(Device&& device) noexcept;

  /**
   * @brief A Device can either be a GPU or a CPU.
   */
  enum Type { GPU, CPU };

  /**
   * @brief Returns the id of the Device that was set with the constructor.
   */
  [[nodiscard]] uint32_t get_id() const;

  /**
   * @brief Returns a const reference to the underlying cl::Device.
   */
  [[nodiscard]] const cl::Device& get_cl_device() const;
  /**
   * @brief Returns a reference to the underlying cl::Device.
   * @return
   */
  cl::Device& get_cl_device();

  /**
   * @brief Returns the name of the device.
   */
  [[nodiscard]] std::string name() const;

  /**
   * @brief Returns the vendor of the device.
   */
  [[nodiscard]] std::string vendor() const;

  /**
   * @brief Returns the driver version of the device.
   */
  [[nodiscard]] std::string driver_version() const;

  /**
   * @brief Returns the OpenCL C version of the device.
   */
  [[nodiscard]] std::string opencl_c_version() const;

  /**
   * @brief Returns the size of the memory in Bytes.
   */
  [[nodiscard]] uint64_t memory_Bytes() const;

  /**
   * @brief Returns the amount of memory used in Bytes.
   *
   *        This value is only correct, if the mcl::Memory object was used for memory management and if this Device was
   *        correctly passed to the mcl::Memory instance.
   */
  [[nodiscard]] uint64_t memory_used_Bytes() const;

  /**
   * @brief Returns the size of the global cache in Bytes.
   */
  [[nodiscard]] uint64_t global_cache_Bytes() const;

  /**
   * @brief Returns the size of the local cache in Bytes.
   */
  [[nodiscard]] uint64_t local_cache_Bytes() const;

  /**
   * @brief Returns the size of the global buffer in Bytes.
   */
  [[nodiscard]] uint64_t max_global_buffer_Bytes() const;

  /**
   * @brief Returns the size of the constant buffer in Bytes.
   */
  [[nodiscard]] uint64_t max_constant_buffer_Bytes() const;

  /**
   * @brief Returns the number of compute units of the device.
   *
   *        Note that each compute unit can have different number of cores. In order to get the actual number of cores,
   *        you should call Device::cores().
   */
  [[nodiscard]] uint64_t compute_units() const;

  /**
   * @brief Returns the total number of cores of the device.
   */
  [[nodiscard]] uint64_t cores() const;

  /**
   * @brief Returns the clock frequency in Hz (ticks per second).
   */
  [[nodiscard]] uint64_t clock_frequency_MHz() const;

  /**
   * @brief Returns the type of the device (either GPU or CPU) as Device::Type.
   */
  [[nodiscard]] Type type() const;

  /**
   * @brief Returns the native vector width for double size floating point numbers.
   *
   *        Returns 0 if the device does not support operating on double size floating points.
   */
  [[nodiscard]] uint64_t fp64() const;

  /**
   * @brief Returns the native vector width for floating point numbers.
   */
  [[nodiscard]] uint64_t fp32() const;

  /**
   * @brief Returns the native vector width for half size floating point numbers.
   *
   *        Returns 0 if the device does not support operating on half size floating points.
   */
  [[nodiscard]] uint64_t fp16() const;

  /**
   * @brief Returns the native vector width for 64 bit integers.
   */
  [[nodiscard]] uint64_t int64() const;

  /**
   * @brief Returns the native vector width for 32 bit integers.
   */
  [[nodiscard]] uint64_t int32() const;

  /**
   * @brief Returns the native vector width for 16 bit integers.
   */
  [[nodiscard]] uint64_t int16() const;

  /**
   * @brief Returns the native vector width for 8 bit integers.
   */
  [[nodiscard]] uint64_t int8() const;

  /**
   * @brief Returns the estimated amount of floating point operations per second in FLOPS/second.
   *
   *        This value is estimated using the number of cores, the instructions per cycle and the frequency of the
   *        device. Therefore, the real FLOPS/second value may differ.
   */
  [[nodiscard]] uint64_t estimated_flops() const;

  [[nodiscard]] bool intel_gt_4gb_buffer_required() const;

 private:
  uint64_t _compute_cores();

  /// set by constructor
  cl::Device _cl_device;

  /// set by constructor
  uint32_t _id;
  /// set by constructor
  uint32_t _instructions_per_cycle;
  /// set by constructor via _compute_cores()
  uint64_t _cores;
  /// set by mcl::Memory
  uint64_t _memory_used_Bytes{0};

  /// set by _compute_cores()
  bool _intel_gt_4gb_buffer_required{false};

  /// used by _compute_cores()
  const std::vector<std::string> nvidia_192{"gt 6", "gt 7", "gtx 6", "gtx 7", "quadro k", "tesla k"};
  /// used by _compute_cores()
  const std::vector<std::string> nvidia_64{"p100",    "v100",      "a100",     "a30",     " 16",       " 20",
                                           "titan v", "titan rtx", "quadro t", "tesla t", "quadro rtx"};
};

/**
 * @brief Provide a naive ostream output for a Device.
 *
 * format: "[GPU|CPU]: <name> (<id>, <vendor>)"
 */
std::ostream& operator<<(std::ostream& os, const Device& device);

std::ostream& operator<<(std::ostream& os, const Device::Type& type);

// ===== DeviceManager =================================================================================================
/**
 * @brief enum values used to retrieve specific devices using the mcl::DeviceManager
 */
enum class Filter {
  MAX_MEMORY,  // Device with most memory
  MIN_MEMORY,  // Device with the smallest memory
  MAX_FLOPS,   // Device with most estimated FLOPS
  MIN_FLOPS,   // Device with least estimated FLOPS
  GPU,         // All GPU devices
  CPU,         // ALL CPU devices
  ID,          // Device by ID
  ALL          // All Devices
};

class DeviceManager {
 public:
  /**
   * @brief Used to retrieve one specific device.
   *
   *        This method is implemented for
   *        T = MAX_MEMORY
   *            MIN_MEMORY
   *            MAX_FLOPS
   *            MIN_FLOPS
   */
  template <Filter T>
  static Device* get();

  /**
   * @brief Used to retrieve multiple devices.
   *
   *        This method is implemented for
   *        T = GPU
   *            CPU
   *            ALL
   */
  template <Filter T>
  static std::vector<Device*> get_list();

  /**
   * @brief Used to retrieve one specific device by id (= value).
   *
   *        This method is implemented for
   *        T = ID
   */
  template <Filter T>
  static Device* get(uint32_t value);

 private:
  DeviceManager();
  static DeviceManager& get_instance();

  std::vector<Device> _devices;
};

// ===== Environment ===================================================================================================
// forward declaration
class Kernel;

class Environment {
  template <typename T>
  friend class Memory;
  friend class Kernel;

 public:
  Environment();
  explicit Environment(Device& device);
  explicit Environment(Device* device);

  Kernel add_kernel(uint32_t range, std::string name, const std::string& cl_c_source);

 private:
  void _init();

  cl::Context _cl_context{};
  Device* _device;
  cl::CommandQueue _cl_queue{};
  cl::Program _cl_program{};
};

// ===== Memory ========================================================================================================

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

  void _allocate_device_buffer() {
    _environment->_device->_memory_used_Bytes += mem_size();
    if (_environment->_device->memory_used_Bytes() > _environment->_device->memory_Bytes()) {
      std::cerr << "Warning: There may be not enough memory available for allocations of this size." << std::endl;
    }
    int error = 0;
    if (_environment->_device->intel_gt_4gb_buffer_required()) {
      // https://github.com/intel/compute-runtime/blob/master/programmers-guide/ALLOCATIONS_GREATER_THAN_4GB.md
      _device_buffer =
          cl::Buffer(_environment->_cl_context, CL_MEM_READ_WRITE|((int)1<<23), mem_size(), nullptr, &error);
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

class Kernel {
  friend class Environment;

 public:
  void set_range(uint64_t range);

  [[nodiscard]] uint64_t range() const;

  template <typename... T>
  void set_parameters(const T&... parameters) {
    link_parameters(parameters...);
  }

  void enqueue_run(unsigned t = 1, const std::vector<cl::Event>* event_waitlist = nullptr,
                   cl::Event* returned_event = nullptr);

  void run(unsigned t = 1, const std::vector<cl::Event>* event_waitlist = nullptr, cl::Event* event_returned = nullptr);

  void finish_queue();

 private:
  Kernel(Environment& environment, uint64_t range, std::string name, const std::string& cl_c_source = "");

  template <typename T>
  void link_parameter(const Memory<T>& memory) {
    _cl_kernel.setArg(_parameter_count++, memory.get_cl_buffer());
  }

  template <typename T>
  void link_parameter(const T& parameter) {
    _cl_kernel.setArg(_parameter_count++, sizeof(T), static_cast<void*>(parameter));
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
  uint64_t _kernel_range;
  cl::NDRange _cl_global_range;
  cl::NDRange _cl_local_range;
  uint64_t _parameter_count{0};

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