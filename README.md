# Windows release

It is enough to download 'dist' directory which includes test signals. Double-click 'gui.exe' to run the application, input a signal audio file and select a decoder.

// pyinstaller - from project folder (windows)

// multiple C++ files (8 decoders)

cmake -S . -B build
cmake --build build --config Release

// create ONEFOLDER executables (dist folder)

pyinstaller ^
--noconsole ^
--name gui ^
--add-binary "build\Release\abs_val.exe;." ^
--add-binary "build\Release\hilbertFIR.exe;." ^
--add-binary "build\Release\hilbertFFT.exe;." ^
--add-binary "build\Release\contrast.exe;." ^
--add-binary "build\Release\falsecolour.exe;." ^
--add-binary "build\Release\pseudocolour1.exe;." ^
--add-binary "build\Release\pseudocolour2.exe;." ^
ui_skin.py

---

// configure cmake

cmake -S . -B build `  -DCMAKE_BUILD_TYPE=Release`
-DCMAKE_TOOLCHAIN_FILE="C:/Users/Korisnik/FOI/Diplomski/software/cmake/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" `
-DVCPKG_TARGET_TRIPLET=x64-windows

// build

cmake --build build --config Release

// run

.\build\Release\noaa_apt.exe "C:\path\to\input.wav" "C:\path\to\out.png" 1200

.\build\Release\noaa_apt.exe ".\test.wav" ".\out.png" 1200 manual pseudo

args: 1 2 3 4 5

width default=1200; mode arg kept for compatibility ("manual"/"opencv", not used in pseudo)

set last arg to "pseudo" to enable the composite

For FTT, install from within vcpkg:

.\vcpkg install fftw3[core]:x64-windows

Add to CMakeLists.txt:

find_package(FFTW3 REQUIRED)
target_link_libraries(noaa_apt PRIVATE SndFile::sndfile FFTW3::fftw3)

Then configure cmake again before building (commands at the top).

For samplerate.h (Better sampling):

.\vcpkg install libsamplerate:x64-windows

Add to CMakeLists.txt:

find_package(Samplerate REQUIRED)
target_link_libraries(noaa_apt PRIVATE Samplerate::samplerate)
