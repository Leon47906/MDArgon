cmake_minimum_required(VERSION 3.10)

# Project name
project(MDArgon)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)

FetchContent_MakeAvailable(json)
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
# Add executable
add_executable(verlet verlet.cpp)

add_executable(mc monte_carlo.cpp)

target_link_libraries(verlet PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(mc PRIVATE nlohmann_json::nlohmann_json)
