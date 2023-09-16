add_library(device_static STATIC device.cpp)
target_link_libraries(device PUBLIC OpenCL::OpenCL)

add_library(device SHARED device.cpp)
target_link_libraries(device PUBLIC OpenCL::OpenCL)

file(GLOB SRC "*.cpp")

add_library(miss-opencl_static STATIC ${SRC})
target_link_libraries(miss-opencl PUBLIC OpenCL::OpenCL device)

add_library(miss-opencl SHARED ${SRC})
target_link_libraries(miss-opencl PUBLIC OpenCL::OpenCL device)