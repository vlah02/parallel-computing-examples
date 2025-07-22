#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdlib>

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

void getOutputBase(const char *root, char *base, size_t len) {
    const char *slash = strrchr(root, '/');
    const char *name = slash ? slash+1 : root;
    strncpy(base, name, len-1);
    base[len-1] = '\0';
    if (char *dot = strchr(base, '.')) *dot = '\0';
}

bool loadSequentialResult(const char *base, int &rows, int &cols, std::vector<float> &C_seq) {
    char seqout[512];
    snprintf(seqout, sizeof(seqout), "output/seq/%s.txt", base);
    return readColMajorMatrixFile(seqout, rows, cols, C_seq);
}

bool loadSequentialTiming(const char *base, double &cpu_sec) {
    char seqtime[512];
    snprintf(seqtime, sizeof(seqtime), "output/seq/%s.txt_time.txt", base);
    FILE *fs = fopen(seqtime, "r");
    if (!fs) return false;
    double sum = 0, tv;
    int cnt = 0;
    while (fscanf(fs, "%lf", &tv) == 1) {
        sum += tv;
        cnt++;
    }
    fclose(fs);
    if (cnt == 0) return false;
    cpu_sec = sum / cnt;
    return true;
}

bool compareResults(const std::vector<float> &a, const std::vector<float> &b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (fabsf(a[i] - b[i]) > tol) return false;
    }
    return true;
}

bool appendTiming(const char *root, double time_sec) {
    char tf[256];
    snprintf(tf, sizeof(tf), "%s_time.txt", root);
    FILE *f = fopen(tf, "a");
    if (!f) {
        perror("fopen timefile");
        return false;
    }
    fprintf(f, "%.6f\n", time_sec);
    fclose(f);
    return true;
}