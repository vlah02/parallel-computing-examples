#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>

bool readColMajorMatrixFile(const char *filename, int &rows, int &cols, std::vector<float> &matrix);
bool writeColMajorMatrixFile(const char *filename, int rows, int cols, const std::vector<float> &matrix);

extern std::string red;
extern std::string green;
extern std::string blue;
extern std::string clear;

#endif // COMMON_H
