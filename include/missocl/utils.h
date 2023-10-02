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

#include <string>
#include <chrono>
#include <stdexcept>

#include <CL/opencl.hpp>

namespace mcl {

inline constexpr const char* cl_error(int error_code);

class OpenCLError : public std::exception {
 public:
  explicit OpenCLError(const char* message) : _message(message) {}

  [[nodiscard]] const char* what() const noexcept override {
    return _message.c_str();
  }

 private:
  std::string _message;
};

void check_opencl_error(cl_int error);

class Timer {
  typedef std::chrono::high_resolution_clock clock;
 public:
  Timer() = default;

  void start() {
    _start = clock::now();
  }

  std::chrono::duration<double> stop() {
    _end = clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(_end - _start);
  }

 private:
   std::chrono::time_point<clock> _start;
   std::chrono::time_point<clock> _end;
};

}