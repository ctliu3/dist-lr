#ifndef DISTLR_DATA_ITER_H_
#define DISTLR_DATA_ITER_H_

#include <string>
#include <vector>
#include <sstream>
#include <fstream>

#include "sample.h"
#include "util.h"

namespace distlr {

class DataIter {
public:
  explicit DataIter(std::string filename, int num_feature_dim)
    : filename_(filename), num_feature_dim_(num_feature_dim), offset_(0),
      round_end_(false) {
      std::ifstream input(filename_.c_str());
      std::string line, buf;
      std::vector<float> feature;
      samples_.clear();

      while (std::getline(input, line)) {
        std::istringstream in(line);
        in >> buf;
        int label = ToInt(buf) == 1 ? 1 : 0;
        feature = std::vector<float>(num_feature_dim_, 0);
        while (in >> buf) {
          auto ss = Split(buf, ':');
          feature[ToInt(ss[0]) - 1] = ToFloat(ss[1]);
        }
        samples_.push_back(Sample(feature, label));
        // std::cout << samples_.back().DebugInfo() << std::endl;
      }
  }
  virtual ~DataIter() {
  }

  // batch_size = -1 means all samples
  std::vector<Sample> NextBatch(int batch_size=100) {
    if (batch_size < 0) {
      batch_size = (int)samples_.size();
    }

    std::vector<Sample> batch;
    for (int i = 0; i < batch_size; ++i) {
      batch.push_back(samples_[offset_]);
      ++offset_;
      if (offset_ == (int)samples_.size()) {
        offset_ = 0;
        round_end_ = true;
      }
    }
    return batch;
  }

  bool HasNext() {
    return !round_end_;
  }

private:
  std::string filename_;
  int num_feature_dim_;
  int offset_;
  bool round_end_;
  std::vector<Sample> samples_;
};

}  // namespace distlr

#endif  // DISTLR_DATA_ITER_H_
