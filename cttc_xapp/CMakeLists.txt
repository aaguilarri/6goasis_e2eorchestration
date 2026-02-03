# Find pkg-config
find_package(PkgConfig REQUIRED)


# Find glib-2.0 and librdkafka using pkg-config
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(RDKAFKA REQUIRED rdkafka)

add_executable(xapp_cttc
                xapp_cttc.c
                common.c
				${CMAKE_SOURCE_DIR}/src/util/alg_ds/alg/defer.c
				${CMAKE_SOURCE_DIR}/src/util/e2ap_ngran_type.c
				)

# Include directories from pkg-config (equivalent to CFLAGS)
include_directories(${GLIB_INCLUDE_DIRS} ${RDKAFKA_INCLUDE_DIRS})

# Link directories (usually not needed since CMake can handle this, but in case)
link_directories(${GLIB_LIBRARY_DIRS} ${RDKAFKA_LIBRARY_DIRS})

# Include directories for Jansson
include_directories(${JANSSON_INCLUDE_DIRS})

# if DB
if(XAPP_DB STREQUAL "MYSQL_XAPP")
	include_directories(${LIBMYSQLCLIENT_INCLUDE_DIRS})
endif()

target_link_libraries(xapp_ebrahim
						PUBLIC
						e42_xapp
						$<TARGET_OBJECTS:e2_time_obj>
						$<$<STREQUAL:${XAPP_DB},MYSQL_XAPP>:${LIBMYSQLCLIENT_LIBRARIES}>
						-pthread
						-lsctp
						-ldl
						-ljansson
                        ${GLIB_LIBRARIES}
                        ${RDKAFKA_LIBRARIES}
						)
