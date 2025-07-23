#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <iomanip>

constexpr const char* RED   = "\033[1;31m";
constexpr const char* GREEN = "\033[1;32m";
constexpr const char* BLUE  = "\033[1;36m";
constexpr const char* BOLD  = "\033[1m";
constexpr const char* CLEAR = "\033[0m";

bool readColMajorMatrixFile(const std::string& filename, int& rows, int& cols, std::vector<float>& matrix);
bool writeColMajorMatrixFile(const std::string& filename, int rows, int cols, const std::vector<float>& matrix);

std::string getOutputBase(const std::string& root);

bool loadSequentialResult(const std::string& base, int& rows, int& cols, std::vector<float>& C_seq);
bool loadSequentialTiming(const std::string& base, double& cpu_sec);

bool compareResults(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-3f);

bool appendTiming(const std::string& root, double time_sec);
