cmake_minimum_required(VERSION 3.16)
project(deeppoly-cpp CXX)

# CMake 4.x dropped support for cmake_minimum_required < 3.5 in subprojects.
# This allows older vendored CMakeLists (e.g. raylib 5.0) to still configure.
set(CMAKE_POLICY_VERSION_MINIMUM "3.5")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Dependencies ──────────────────────────────────────────────────────────────
include(FetchContent)

# Raylib
FetchContent_Declare(raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        5.5
    GIT_SHALLOW    TRUE
)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(raylib)

# tiny-cpp-nn — header-only, no CMakeLists; use Populate + INTERFACE target
FetchContent_Declare(tinycppnn
    GIT_REPOSITORY https://github.com/jeetrex17/tiny-cpp-nn.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(tinycppnn)
if(NOT tinycppnn_POPULATED)
    FetchContent_Populate(tinycppnn)
endif()
add_library(tinycppnn INTERFACE)
target_include_directories(tinycppnn INTERFACE ${tinycppnn_SOURCE_DIR})

# ── Targets ───────────────────────────────────────────────────────────────────
add_executable(train  src/train.cpp)
add_executable(attack src/attack.cpp)
add_executable(verify src/verify.cpp)
add_executable(show   src/show.cpp)

foreach(tgt train attack verify show)
    target_include_directories(${tgt} PRIVATE include)
    target_link_libraries(${tgt} PRIVATE tinycppnn)
    target_compile_options(${tgt} PRIVATE -O2)
endforeach()

target_link_libraries(show PRIVATE raylib)

if(APPLE)
    target_link_libraries(show PRIVATE
        "-framework OpenGL"
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
        "-framework AudioToolbox"
    )
endif()

# ── Output dirs (created relative to build dir at configure time) ─────────────
foreach(dir data models output)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${dir})
endforeach()
