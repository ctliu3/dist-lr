#include "ps/ps.h"
#include "cmath"
#include "lr.h"
#include "util.h"
#include "sample.h"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace distlr {

LR::LR(int num_feature_dim, float learning_rate, float C, int random_state)
   : num_feature_dim_(num_feature_dim), learning_rate_(learning_rate), C_(C),
     random_state_(random_state) {
  InitWeight_();
}

void LR::SetKVWorker(ps::KVWorker<float>* kv) {
  kv_ = kv;
}

void LR::Train(DataIter& iter, int batch_size = 100) {
  while (iter.HasNext()) {
    std::vector<Sample> batch = iter.NextBatch(batch_size);

    PullWeight_();

    std::vector<float> grad(weight_.size());
    for (size_t j = 0; j < weight_.size(); ++j) {
      grad[j] = 0;
      for (size_t i = 0; i < batch.size(); ++i) {
        auto& sample = batch[i];
        grad[j] += (Sigmoid_(sample.GetFeature()) - sample.GetLabel()) * sample.GetFeature(j);
      }
      grad[j] = 1. * grad[j] / batch.size() + C_ * weight_[j] / batch.size();
    }

    PushGradient_(grad);
  }
}

void LR::Test(DataIter& iter, int num_iter) {
  PullWeight_(); // pull the latest weight
  std::vector<Sample> batch = iter.NextBatch(-1);
  float acc = 0;
  for (size_t i = 0; i < batch.size(); ++i) {
    auto& sample = batch[i];
    if (Predict_(sample.GetFeature()) == sample.GetLabel()) {
      ++acc;
    }
  }
  time_t rawtime;
  time(&rawtime);
  struct tm* curr_time = localtime(&rawtime);
  std::cout << std::setw(2) << curr_time->tm_hour << ':' << std::setw(2)
    << curr_time->tm_min << ':' << std::setw(2) << curr_time->tm_sec
    << " Iteration "<< num_iter << ", accuracy: " << acc / batch.size()
    << std::endl;
}

std::vector<float> LR::GetWeight() {
  return weight_;
}

ps::KVWorker<float>* LR::GetKVWorker() {
  return kv_;
}

bool LR::SaveModel(std::string& filename) {
  std::ofstream fout(filename.c_str());
  fout << num_feature_dim_ << std::endl;
  for (int i = 0; i < num_feature_dim_; ++i) {
    fout << weight_[i] << ' ';
  }
  fout << std::endl;
  fout.close();
  return true;
}

std::string LR::DebugInfo() {
  std::ostringstream out;
  for (size_t i = 0; i < weight_.size(); ++i) {
    out << weight_[i] << " ";
  }
  return out.str();
}

void LR::InitWeight_() {
  srand(random_state_);
  weight_.resize(num_feature_dim_);
  for (size_t i = 0; i < weight_.size(); ++i) {
    weight_[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}

int LR::Predict_(std::vector<float> feature) {
  float z = 0;
  for (size_t j = 0; j < weight_.size(); ++j) {
    z += weight_[j] * feature[j];
  }
  return z > 0;
}

float LR::Sigmoid_(std::vector<float> feature) {
  float z = 0;
  for (size_t j = 0; j < weight_.size(); ++j) {
    z += weight_[j] * feature[j];
  }
  return 1. / (1. + exp(-z));
}

void LR::PullWeight_() {
  std::vector<ps::Key> keys(num_feature_dim_);
  std::vector<float> vals;
  for (int i = 0; i < num_feature_dim_; ++i) {
    keys[i] = i;
  }
  kv_->Wait(kv_->Pull(keys, &vals));
  weight_ = vals;
}

void LR::PushGradient_(const std::vector<float>& grad) {
  std::vector<ps::Key> keys(num_feature_dim_);
  for (int i = 0; i < num_feature_dim_; ++i) {
    keys[i] = i;
  }
  kv_->Wait(kv_->Push(keys, grad));
}

} // namespace distlr
