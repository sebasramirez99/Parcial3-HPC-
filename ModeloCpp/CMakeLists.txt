cmake_minimum_required(VERSION 3.5)

project(ModeloCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(project_sources
    main.cpp
    REGRESIONLINEAL/regresion_lineal.cpp
    EXTRACCION/extracciondata.cpp)

set(project_headers
    REGRESIONLINEAL/regresion_lineal.h
    EXTRACCION/extracciondata.h)


add_executable(${PROJECT_NAME}
                ${project_sources}
                ${project_headers})
