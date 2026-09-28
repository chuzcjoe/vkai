#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vkai {

struct WeightTensor {
  std::vector<uint32_t> shape;
  std::vector<float> data;
};

class Weights {
 public:
  const std::string& name() const { return name_; }
  const WeightTensor* Find(std::string_view tensor_name) const;
  bool Contains(std::string_view tensor_name) const;

 private:
  friend class WeightsLoader;

  std::string name_;
  std::unordered_map<std::string, WeightTensor> tensors_;
};

class WeightsLoader {
 public:
  static bool Load(const std::string& filename, Weights& weights,
                   std::string* error_message = nullptr);
};

}  // namespace vkai
