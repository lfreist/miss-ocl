add_library(device_static STATIC device.cpp utils.cpp)
target_link_libraries(device_static PUBLIC OpenCL::HeadersCpp OpenCL::OpenCL)
target_include_directories(device_static PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(device SHARED device.cpp utils.cpp)
target_link_libraries(device PUBLIC OpenCL::HeadersCpp OpenCL::OpenCL)
target_include_directories(device PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(mcl:device ALIAS device)
add_library(mcl:device_static ALIAS device_static)

file(GLOB SRC "*.cpp")

add_library(miss-opencl_static STATIC ${SRC})
target_link_libraries(miss-opencl_static PUBLIC OpenCL::HeadersCpp OpenCL::OpenCL device_static)
target_include_directories(miss-opencl_static PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(miss-opencl SHARED ${SRC})
target_link_libraries(miss-opencl PUBLIC OpenCL::HeadersCpp OpenCL::OpenCL device)
target_include_directories(miss-opencl PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(mcl:opencl ALIAS miss-opencl)
add_library(mcl:opencl_static ALIAS miss-opencl_static)