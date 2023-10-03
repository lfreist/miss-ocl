add_library(device_static STATIC device.cpp utils.cpp)
target_link_libraries(device_static PUBLIC OpenCL::OpenCL)

add_library(device SHARED device.cpp utils.cpp)
target_link_libraries(device PUBLIC OpenCL::OpenCL)

file(GLOB SRC "*.cpp")

add_library(miss-opencl_static STATIC ${SRC})
target_link_libraries(miss-opencl_static PUBLIC OpenCL::OpenCL device_static)

add_library(miss-opencl SHARED ${SRC})
target_link_libraries(miss-opencl PUBLIC OpenCL::OpenCL device)