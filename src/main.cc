#include <iostream>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>
#include "ps/ps.h"

#include "lr.h"
#include "util.h"
#include "data_iter.h"

const int kSyncMode = -1;

template <typename Val>
class KVStoreDistServer {
public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<float>(0);
    ps_server_->set_request_handle(
      std::bind(&KVStoreDistServer::DataHandle, this, _1, _2, _3));

    sync_mode_ = !strcmp(ps::Environment::Get()->find("SYNC_MODE"), "1");
    learning_rate_ = distlr::ToFloat(ps::Environment::Get()->find("LEARNING_RATE"));

    std::string mode = sync_mode_ ? "sync" : "async";
    std::cout << "Server mode: " << mode << std::endl;
  }

  ~KVStoreDistServer() {
    if (ps_server_) {
      delete ps_server_;
    }
  }

private:

  // threadsafe
  void DataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<Val>& req_data,
                  ps::KVServer<Val>* server) {
    int key = DecodeKey(req_data.keys[0]);
    auto& weights = weights_[key];

    size_t n = req_data.keys.size();
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
      if (weights.empty()) {
        std::cout << "Init weight" << std::endl;
        weights.resize(n);
        for (int i = 0; i < n; ++i) {
          weights[i] = req_data.vals[i];
        }
        server->Response(req_meta);
      } else if (sync_mode_) {
        auto& merged = merge_buf_[key];
        if (merged.vals.empty()) {
          merged.vals.resize(n, 0);
        }

        for (int i = 0; i < n; ++i) {
          merged.vals[i] += req_data.vals[i];
        }

        merged.request.push_back(req_meta);
        if (merged.request.size() == (size_t)ps::NumWorkers()) {
          // update the weight
          for (size_t i = 0; i < n; ++i) {
            weights[i] -= learning_rate_ * req_data.vals[i] / merged.request.size();
          }
          for (const auto& req : merged.request) {
            server->Response(req);
          }
          merged.request.clear();
          merged.vals.clear();
        }
      } else { // async push
        for (size_t i = 0; i < n; ++i) {
          weights[i] -= learning_rate_ * req_data.vals[i];
        }
        server->Response(req_meta);
      }
    } else { // pull
      CHECK(!weights_.empty()) << "init " << key << " first";

      ps::KVPairs<Val> response;
      response.keys = req_data.keys;
      response.vals.resize(n);
      for (size_t i = 0; i < n; ++i) {
        response.vals[i] = weights[i];
      }
      server->Response(req_meta, response);
    }
  }

  int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }

  bool sync_mode_;
  float learning_rate_;

  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    std::vector<Val> vals;
  };

  std::unordered_map<int, std::vector<Val>> weights_;
  std::unordered_map<int, MergeBuf> merge_buf_;
  ps::KVServer<float>* ps_server_;
};

void StartServer() {
  if (!ps::IsServer()) {
    return;
  }
  auto server = new KVStoreDistServer<float>();
  ps::RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
  if (!ps::IsWorker()) {
    return;
  }

  std::string root = ps::Environment::Get()->find("DATA_DIR");
  int num_feature_dim =
    distlr::ToInt(ps::Environment::Get()->find("NUM_FEATURE_DIM"));

  int rank = ps::MyRank();
  ps::KVWorker<float>* kv = new ps::KVWorker<float>(0);
  distlr::LR lr = distlr::LR(num_feature_dim);
  lr.SetKVWorker(kv);

  if (rank == 0) {
    auto vals = lr.GetWeight();
    std::vector<ps::Key> keys(vals.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = i;
    }
    kv->Wait(kv->Push(keys, vals));
  }
  ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);

  std::cout << "Worker[" << rank << "]: start working..." << std::endl;
  int num_iteration = distlr::ToInt(ps::Environment::Get()->find("NUM_ITERATION"));
  int batch_size = distlr::ToInt(ps::Environment::Get()->find("BATCH_SIZE"));
  int test_interval = distlr::ToInt(ps::Environment::Get()->find("TEST_INTERVAL"));

  for (int i = 0; i < num_iteration; ++i) {
    std::string filename = root + "/train/part-00" + std::to_string(rank + 1);
    distlr::DataIter iter(filename, num_feature_dim);
    lr.Train(iter, batch_size);

    if (rank == 0 and i % test_interval == 0) {
      std::string filename = root + "/test/part-001";
      distlr::DataIter test_iter(filename, num_feature_dim);
      lr.Test(test_iter, i);
    }
  }
  std::string modelfile = root + "/models/part-00" + std::to_string(rank + 1);
  lr.SaveModel(modelfile);
}

int main() {
  StartServer();

  ps::Start();
  RunWorker();

  ps::Finalize();
  return 0;
}
