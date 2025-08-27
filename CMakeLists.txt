cmake_minimum_required(VERSION 3.16)
project(noaa_apt_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(SndFile REQUIRED)  # via vcpkg or system package

add_executable(noaa_apt noaa_apt.cpp)
target_link_libraries(noaa_apt PRIVATE SndFile::sndfile)
target_include_directories(noaa_apt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}) # for stb_image_write.h
