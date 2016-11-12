#ifndef DISTLR_SAMPLE_H_
#define DISTLR_SAMPLE_H_

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

namespace distlr {

class Sample {
public:
  explicit Sample(int num_feature_dim): num_feature_dim_(num_feature_dim) {
    feature_.resize(num_feature_dim);
  }
  explicit Sample(std::vector<float>& feature, int label)
    : feature_(feature), label_(label) {
  }
  virtual ~Sample() {
  }

  void SetLabel(int label) {
    label_ = label;
  }

  void SetFeatures(const std::vector<float>& feature) {
    feature_ = feature;
  }

  std::pair<std::vector<float>, int> GetSample() {
    return std::make_pair(feature_, label_);
  }

  std::vector<float> GetFeature() {
    return feature_;
  }

  float GetFeature(int index) {
    return feature_[index];
  }

  int GetLabel() {
    return label_;
  }

  std::string DebugInfo() {
    std::string str = std::to_string(label_);
    for (int i = 0; i < (int)feature_.size(); ++i) {
      if (feature_[i]) {
        str += " " + std::to_string(i) + ":" + std::to_string(feature_[i]);
      }
    }
    return str;
  }

private:
  int num_feature_dim_;
  std::vector<float> feature_;
  int label_;
};

}  // namespace distlr

#endif  // DISTLR_SAMPLE_H_
