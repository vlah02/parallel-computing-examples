#include "../include/common.hpp"

bool readColMajorMatrixFile(const std::string& fn, int& nr_row, int& nr_col, std::vector<float>& v) {
    std::ifstream f(fn);
    if (!f.is_open()) return false;

    f >> nr_row >> nr_col;
    float data;
    while (f >> data)
        v.push_back(data);

    return true;
}

bool writeColMajorMatrixFile(const std::string& fn, int nr_row, int nr_col, const std::vector<float>& v) {
    std::ofstream f(fn);
    if (!f.is_open()) return false;

    f << nr_row << " " << nr_col << " ";
    for (float val : v)
        f << val << ' ';
    f << "\n";
    return true;
}

std::string getOutputBase(const std::string& root) {
    size_t slash = root.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? root : root.substr(slash + 1);

    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos)
        base = base.substr(0, dot);

    return base;
}

bool loadSequentialResult(const std::string& base, int& rows, int& cols, std::vector<float>& C_seq) {
    return readColMajorMatrixFile("output/seq/" + base + ".txt", rows, cols, C_seq);
}

bool loadSequentialTiming(const std::string& base, double& cpu_sec) {
    std::ifstream fs("output/seq/" + base + "_time.txt");
    if (!fs.is_open()) return false;
    double sum = 0, tv;
    int cnt = 0;
    while (fs >> tv) {
        sum += tv;
        ++cnt;
    }
    if (cnt == 0) return false;
    cpu_sec = sum / cnt;
    return true;
}

bool compareResults(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

bool appendTiming(const std::string& root, double time_sec) {
    std::string timing_file = root;
    size_t dot = timing_file.rfind(".txt");
    if (dot != std::string::npos) {
        timing_file.replace(dot, 4, "_time.txt");
    } else {
        std::cerr << "appendTiming: Output file path does not contain .txt extension: \""
                  << root << "\"" << std::endl;
        return false;
    }

    std::ofstream f(timing_file, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "appendTiming: Failed to open timing file \""
                  << timing_file << "\" for append." << std::endl;
        return false;
    }
    f << std::fixed << std::setprecision(6) << time_sec << "\n";
    return true;
}
