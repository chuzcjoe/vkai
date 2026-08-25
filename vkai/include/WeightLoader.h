#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vkai {

struct ModelWeights {
  std::vector<float> conv1_weights;
  std::vector<float> conv1_bias;
  std::vector<float> conv2_weights;
  std::vector<float> conv2_bias;
  std::vector<float> fc1_weights;
  std::vector<float> fc1_bias;
  std::vector<float> fc2_weights;
  std::vector<float> fc2_bias;
  std::vector<float> fc3_weights;
  std::vector<float> fc3_bias;
};

class WeightLoader {
public:
  static bool Load(const std::string &filename, ModelWeights &weights);
};

} // namespace vkai
