#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
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
  static ModelWeights Load(const std::string &filename);
};

} // namespace vkai