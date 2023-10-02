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

#include <missocl/device.h>
#include <missocl/environment.h>
#include <missocl/kernel.h>
#include <missocl/memory.h>
#include <missocl/utils.h>

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#define WORKGROUP_SIZE 64
#define CL_HPP_TARGET_OPENCL_VERSION 300

#define KERNEL_CODE(name, ...) std::string name(#__VA_ARGS__)