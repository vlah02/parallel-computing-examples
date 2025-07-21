#include <fstream>
#include <iostream>

#include "../include/common.hpp"

bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float> &v) {
//    std::cerr << "Opening file: " << fn << std::endl;
    std::ifstream f(fn);
    if (!f.good()) return false;

    f >> nr_row >> nr_col;
    float data;
//    std::cerr << "Matrix dimension: " << nr_row << "x" << nr_col << std::endl;
    while (f >> data) {
        v.push_back(data);
    }

    return true;
}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, const std::vector<float> &v) {
//    std::cerr << "Opening file: " << fn << " for write." << std::endl;
    std::ofstream f(fn);
    if (!f.good()) return false;

    f << nr_row << " " << nr_col << " ";
//    std::cerr << "Matrix dimension: " << nr_row << "x" << nr_col << std::endl;
    for (float val : v) {
        f << val << ' ';
    }
    f << "\n";
    return true;
}
