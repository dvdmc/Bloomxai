cmake_minimum_required(VERSION 3.10)
project(Bloomxai)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")

  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)

#---- Add the subdirectory cmake ----

find_package(PCL QUIET COMPONENTS common io)

find_package(benchmark QUIET)
find_package(octomap QUIET)
find_package(Eigen3 QUIET)

#---- Use Address sanitizer if compiled in Debug mode ----
option(BONXAI_SANITIZE "Add address sanitizer when compiling in Debug mode" OFF)

if(BONXAI_SANITIZE)
    set (CMAKE_CXX_DEBUG_FLAGS "${CMAKE_CXX_DEBUG_FLAGS}  -fsanitize=address")
    set (CMAKE_LINKER_DEBUG_FLAGS "${CMAKE_LINKER_DEBUG_FLAGS}  -fsanitize=address")
endif()


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fno-omit-frame-pointer)
endif()


####################################################

if(Eigen3_FOUND AND PCL_FOUND )
    add_subdirectory(bloomxai_map)
endif()

# Default behavior
find_package(ament_cmake QUIET)

if ( ament_cmake_FOUND )

    project(bloomxai_ros)

    # Add colcon-specific configuration here
    message(STATUS "Building with colcon")

    find_package(ament_cmake_auto REQUIRED)
    ament_auto_find_build_dependencies()

    rosidl_generate_interfaces(${PROJECT_NAME}
      "bloomxai_ros/srv/LoadMap.srv"
      "bloomxai_ros/srv/SaveMap.srv"
    )

    include_directories(bloomxai_ros/include 3rdparty)

    ament_auto_add_library(bloomxai_server SHARED
      bloomxai_ros/src/bloomxai_server.cpp
    )

    target_link_libraries(bloomxai_server
      bloomxai_map
      ${PCL_LIBRARIES}
    )

    rosidl_get_typesupport_target(cpp_typesupport_target 
    ${PROJECT_NAME} "rosidl_typesupport_cpp")

    target_link_libraries(bloomxai_server "${cpp_typesupport_target}")

    rclcpp_components_register_node(bloomxai_server
      PLUGIN "bloomxai_server::BloomxaiServer"
      EXECUTABLE bloomxai_server_node
    )

    ament_auto_package(
      INSTALL_TO_SHARE
      bloomxai_ros/config
      bloomxai_ros/launch
    )

else()

  message(STATUS "Building with cmake")

endif()
