#pragma once

#include <vector>
#include <string>

bool readColMajorMatrixFile(const char *filename, int &rows, int &cols, std::vector<float> &matrix);
bool writeColMajorMatrixFile(const char *filename, int rows, int cols, const std::vector<float> &matrix);