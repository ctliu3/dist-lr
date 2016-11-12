#ifndef DISTLR_LR_H_
#define DISTLR_LR_H_

#include "data_iter.h"

namespace distlr {

class LR {
public:
  explicit LR(int num_feature_dim, float learning_rate=0.001, float C_=1,
              int random_state=0);
  virtual ~LR() {
    if (kv_) {
      delete kv_;
    }
  }

  void SetKVWorker(ps::KVWorker<float>* kv);

  void Train(DataIter& iter, int num_iter);

  void Test(DataIter& iter, int num_iter);

  std::vector<float> GetWeight();

  ps::KVWorker<float>* GetKVWorker();

  bool SaveModel(std::string& filename);

  std::string DebugInfo();

private:
  void InitWeight_();

  int Predict_(std::vector<float> feature);

  float Sigmoid_(std::vector<float> feature);

  void PullWeight_();

  void PushGradient_(const std::vector<float>& grad);

  int num_feature_dim_;
  float learning_rate_;
  float C_;

  int random_state_;

  std::vector<float> weight_;

  ps::KVWorker<float>* kv_;
};

}  // namespace distlr

#endif  // LR_H_
