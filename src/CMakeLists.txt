add_library(device device.cpp)
target_link_libraries(device PUBLIC OpenCL::OpenCL)

file(GLOB SRC "*.cpp")

add_library(opencl STATIC ${SRC})
target_link_libraries(opencl PUBLIC OpenCL::OpenCL)