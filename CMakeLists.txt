cmake_minimum_required(VERSION 3.16)
project(noaa_apt LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# via vcpkg
find_package(SndFile REQUIRED)
find_package(FFTW3 REQUIRED)                
find_package(SampleRate CONFIG REQUIRED)    
find_package(OpenCV REQUIRED)

# helper to avoid repetition
function(add_decoder target_name src_file out_name)
  add_executable(${target_name} ${src_file})
  target_link_libraries(${target_name}
    PRIVATE
      SndFile::sndfile
      FFTW3::fftw3
      SampleRate::samplerate
      ${OpenCV_LIBS}
  )
  # for stb_image_write.h in source dir (and any local headers)
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  # make the produced .exe be exactly the filename I expect
  set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${out_name})
endfunction()

# NOTE: target names are safe identifiers; OUTPUT_NAME sets the final exe name.
add_decoder(dec_abs_val       abs_val.cpp       abs_val)
add_decoder(dec_cosine        cosine.cpp        cosine)
add_decoder(dec_hilbertFIR    hilbertFIR.cpp    hilbertFIR)
add_decoder(dec_hilbertFFT    hilbertFFT.cpp    hilbertFFT)
add_decoder(dec_contrast      contrast.cpp      contrast)
add_decoder(dec_falsecolour   falsecolour.cpp   falsecolour)
add_decoder(dec_pseudocolour1 pseudocolour1.cpp pseudocolour1)
add_decoder(dec_pseudocolour2 pseudocolour2.cpp pseudocolour2)
