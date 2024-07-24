cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)


project(Sailboat)



pico_sdk_init()

add_executable(sailboat
    main.c

)

pico_enable_stdio_uart(sailboat 0)
pico_enable_stdio_usb(sailboat 1)



target_link_libraries(sailboat pico_stdlib pico_sync pico_multicore hardware_pwm hardware_uart hardware_sync hardware_resets hardware_i2c)

pico_add_extra_outputs(sailboat)