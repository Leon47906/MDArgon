cmake_minimum_required(VERSION 3.10)
project(MDArgon)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mdcore STATIC verlet.cpp)
target_link_libraries(mdcore PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(mdcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(verlet verlet_main.cpp)
target_link_libraries(verlet PRIVATE mdcore)

add_executable(mc monte_carlo.cpp)
target_link_libraries(mc PRIVATE mdcore)
