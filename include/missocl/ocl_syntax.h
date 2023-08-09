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

// definitions for syntax highlighting within kernel code

// OpenCL API 1.2: https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf
// Datatypes
#define char2
#define char3
#define char4
#define char8
#define char16
#define uchar
#define uchar2
#define uchar3
#define uchar4
#define uchar8
#define uchar16

#define short2
#define short3
#define short4
#define short8
#define short16
#define ushort2
#define ushort3
#define ushort4
#define ushort8
#define ushort16

#define int2
#define int3
#define int4
#define int8
#define int16
#define uint2
#define uint3
#define uint4
#define uint8
#define uint16

#define long2
#define long3
#define long4
#define long8
#define long16
#define ulong2
#define ulong3
#define ulong4
#define ulong8
#define ulong16

#define half
#define half2
#define half3
#define half4
#define half8
#define half16

#define float2
#define float3
#define float4
#define float8
#define float16

#define double2
#define double3
#define double4
#define double8
#define double16

// cast
#define as_char2(x)
#define as_char3(x)
#define as_char4(x)
#define as_char8(x)
#define as_char16(x)
#define as_uchar(x)
#define as_uchar2(x)
#define as_uchar3(x)
#define as_uchar4(x)
#define as_uchar8(x)
#define as_uchar16(x)

#define as_short2(x)
#define as_short3(x)
#define as_short4(x)
#define as_short8(x)
#define as_short16(x)
#define as_ushort2(x)
#define as_ushort3(x)
#define as_ushort4(x)
#define as_ushort8(x)
#define as_ushort16(x)

#define as_int2(x)
#define as_int3(x)
#define as_int4(x)
#define as_int8(x)
#define as_int16(x)
#define as_uint2(x)
#define as_uint3(x)
#define as_uint4(x)
#define as_uint8(x)
#define as_uint16(x)

#define as_long2(x)
#define as_long3(x)
#define as_long4(x)
#define as_long8(x)
#define as_long16(x)
#define as_ulong2(x)
#define as_ulong3(x)
#define as_ulong4(x)
#define as_ulong8(x)
#define as_ulong16(x)

#define as_half(x)
#define as_half2(x)
#define as_half3(x)
#define as_half4(x)
#define as_half8(x)
#define as_half16(x)

#define as_float(x)
#define as_float2(x)
#define as_float3(x)
#define as_float4(x)
#define as_float8(x)
#define as_float16(x)

#define as_double(x)
#define as_double2(x)
#define as_double3(x)
#define as_double4(x)
#define as_double8(x)
#define as_double16(x)

// address Space Qualifiers
#define __global
#define global
#define __local
#define local
#define __constant
#define constant
#define __private
#define private

// Function Qualifiers
#define __kernel
#define kernel
#define __attribute__(x)

// Math Constants
#define MAXFLOAT
#define HUGE_VALF
#define HUGE_VAL
#define INFINITY
#define NAN
#define M_E_F
#define M_LOG2E_F
#define M_LOG10E_F
#define M_LN2_F
#define M_LN10_F
#define M_PI_F
#define M_PI_2_F
#define M_PI_4_F
#define M_1_PI_F
#define M_2_PI_F
#define M_2_SQRTPI_F
#define M_SQRT2_F
#define M_SQRT1_2_F

// Integer Built-in Functions
#define abs(x)
#define abs_diff(x, y)
#define add_sat(x, y)
#define hadd(x, y)
#define rhadd(x, y)
#define clamp(x, y, z)
#define clz(x)
#define mad_hi(x, y, z)
#define mad_sat(x, y, z)
#define max(x, y)
#define min(x, y)
#define mul_hi(x, y)
#define rotate(x, y)
#define sub_sat(x, y)
#define popcount(x)
#define upsample(x, y)
#define mad24(x, y, z)
#define mul24(x, y, z)

// Math Built-in Functions
#define acos(x)
#define acosh(x)
#define acospi(x)
#define asin(x)
#define asinh(x)
#define asinpi(x)
#define atan(x)
#define atan2(x, y)
#define atanh(x)
#define atanpi(x)
#define atan2pi(x, y)
#define cbrt(x)
#define ceil(x)
#define copysign(x, y)
#define cos(x)
#define cosh(x)
#define cospi(x)
#define half_divide(x, y)
#define native_divide(x, y)
#define erfc(x)
#define erf(x)
#define exp(x)
#define exp2(x)
#define exp10(x)
#define expm1(x)
#define fabs(x)
#define fdim(x, y)
#define floor(x)
#define fma(x, y, z)
#define fmax(x, y)
#define fmin(x, y)
#define fmod(x, y)
#define fract(x, y)
#define frexp(x, y)
#define hypot(x, y)
#define ilogb(x)
#define ldexp(x, y)
#define lgamma(x)
#define lgamma_r(x, y)
#define log(x)
#define log2(x)
#define log10(x)
#define log1p(x)
#define logb(x)
#define mad(x, y, z)
#define maxmag(x, y)
#define minmag(x, y)
#define modf(x, y)
#define nan(x)
#define nextafter(x, y)
#define pow(x, y)
#define pown(x, y)
#define powr(x, y)
#define half_recip(x)
#define native_recip(x)
#define remainder(x)
#define remquo(x, y, z)
#define rint(x)
#define rootn(x, y)
#define round(x)
#define rsqrt(x)
#define sin(x)
#define sincos(x, y)
#define sinh(x)
#define sinpi(x)
#define sqrt(x)
#define tan(x)
#define tanh(x)
#define tanpi(x)
#define tgamma(x)
#define trunc(x)

// Geometric Built-in Functions
#define cross(x, y)
#define distance(x, y)
#define dot(x, y)
#define length(x)
#define normalize(x)
#define fast_distance(x, y)
#define fast_length(x)
#define fast_normalize(x)

// Vector Data Load/Store
#define vloadn(x, y)
#define vload_half(x, y)
#define vload_halfn(x, y)
#define vstoren(x, y, z)
#define vstore_half(x, y, z)
#define vstore_halfn(x, y, z)
#define vloada_halfn(x, y)
#define vstorea_halfn(x, y, z)

// Async Copies and Prefetch Functions
#define async_work_group_copy(x, y, z, t)
#define async_work_group_strided_copy(x, y, z, t)
#define wait_group_events(x, y)
#define prefetch(x, y)

// Work-Item Built-in Functions
#define get_work_dim()
#define get_global_size(x)
#define get_global_id(x)
#define get_local_size(x)
#define get_local_id(x)
#define get_num_groups(x)
#define get_group_id(x)
#define get_global_offset(x)

// Common Built-in Functions
#define clamp(x, y, z)
#define degrees(x)
#define max(x, y)
#define min(x, y)
#define mix(x, y, z)
#define radians(x)
#define step(x, y)
#define smoothstep(x, y, z)
#define sign(x)

// Relational Built-In-Functions
#define isequal(x, y)
#define isnotequal(x, y)
#define isgreater(x, y)
#define isgreaterequal(x, y)
#define isless(x, y)
#define islessequal(x, y)
#define islessgreater(x, y)
#define isfinite(x)
#define isinf(x)
#define isnan(x)
#define isnormal(x)
#define isordered(x, y)
#define isunordered(x, y)
#define signbit(x)
#define any(x)
#define all(x)
#define bitselect(x, y, z)
#define select(x, y, z)

// Atomic Functions
#define atomic_add(x, y)
#define atomic_sub(x, y)
#define atomic_xchg(x, y)
#define atomic_inc(x, y)
#define atomic_dec(x, y)
#define atomic_cmpxchg(x, y)
#define atomic_min(x, y)
#define atomic_max(x, y)
#define atomic_and(x, y)
#define atomic_or(x, y)
#define atomic_xor(x, y)