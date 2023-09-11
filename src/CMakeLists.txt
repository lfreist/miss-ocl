add_library(device STATIC device.cpp)
target_link_libraries(device PUBLIC OpenCL::OpenCL)

file(GLOB SRC "*.cpp")

add_library(miss-opencl STATIC ${SRC})
target_link_libraries(miss-opencl PUBLIC OpenCL::OpenCL device)