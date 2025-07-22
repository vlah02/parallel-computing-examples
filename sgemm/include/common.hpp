#pragma once

#include <vector>
#include <string>

#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define BLUE    "\033[1;36m"
#define BOLD    "\033[1m"
#define CLEAR   "\033[0m"

bool readColMajorMatrixFile(const char *filename, int &rows, int &cols, std::vector<float> &matrix);
bool writeColMajorMatrixFile(const char *filename, int rows, int cols, const std::vector<float> &matrix);

void getOutputBase(const char *root, char *base, size_t len);

bool loadSequentialResult(const char *base, int &rows, int &cols, std::vector<float> &C_seq);
bool loadSequentialTiming(const char *base, double &cpu_sec);

bool compareResults(const std::vector<float> &a, const std::vector<float> &b, float tol = 1e-3f);

bool appendTiming(const char *root, double time_sec);