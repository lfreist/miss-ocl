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

#include <iomanip>
#include <iostream>

#define fill(width) std::left << std::setw(width)

int main(int argc, char** argv) {
  std::cout << "+------------------------------------------------------------------------------+\n";
  std::cout << "| Available OpenCL Devices                                                     |\n";
  std::cout << "+-----------------------+------------------------------------------------------+\n";
  for (const auto* device : mcl::DeviceManager::get_list<mcl::Filter::ALL>()) {
    std::cout << "| ID                    | " << fill(53) << device->get_id() << std::right << "|\n"
              << "| Name                  | " << fill(53) << device->name() << std::right << "|\n"
              << "| Vendor                | " << fill(53) << device->vendor() << std::right << "|\n"
              << "| Type                  | " << fill(53) << device->type() << std::right << "|\n"
              << "| Memory (MiB)          | " << fill(53) << device->memory_Bytes() / 0x100000 << std::right << "|\n"
              << "| Memory used (MiB)     | " << fill(53) << device->memory_used_Bytes() / 0x100000 << std::right
              << "|\n"
              << "| Clock Frequency (MHz) | " << fill(53) << device->clock_frequency_MHz() << std::right << "|\n"
              << "| Computing Units       | " << fill(53) << device->compute_units() << std::right << "|\n"
              << "| Cores                 | " << fill(53) << device->cores() << std::right << "|\n"
              << "| Estimated FLOPS/s     | " << fill(53) << device->estimated_flops() << std::right << "|\n"
              << "| Driver Version        | " << fill(53) << device->driver_version() << std::right << "|\n"
              << "| OpenCL C Version      | " << fill(53) << device->opencl_c_version() << std::right << "|\n";
    std::cout << "+-----------------------+------------------------------------------------------+\n";
  }

  return 0;
}