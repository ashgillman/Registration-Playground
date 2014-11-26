cmake_minimum_required(VERSION 2.8)

project(RegistrationPlayground)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(RegistrationPlayground RegistrationPlayground.cxx)
add_custom_command(TARGET MyTarget PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/config $<TARGET_FILE_DIR:MyTarget>)

target_link_libraries(RegistrationPlayground ${ITK_LIBRARIES})