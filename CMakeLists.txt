cmake_minimum_required(VERSION 3.16)
project(noaa_apt LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# via vcpkg
find_package(SndFile REQUIRED)
find_package(FFTW3 REQUIRED)                # provides FFTW3::fftw3
find_package(SampleRate CONFIG REQUIRED)    # provides SampleRate::samplerate

add_executable(noaa_apt noaa_apt.cpp)
target_link_libraries(noaa_apt
  PRIVATE
    SndFile::sndfile
    FFTW3::fftw3
    SampleRate::samplerate
)

# for stb_image_write.h in source dir
target_include_directories(noaa_apt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
