#include <gtest/gtest.h>
#include <iostream>

#include "WeightLoader.h"

namespace vkai {
namespace test {

TEST(LoadWeightsTest, MNIST) {
  const auto weights = WeightLoader::Load("../python/mnist_weights.bin");
  std::cout << "fc1_weights size: " << weights.fc1_weights.size() << '\n';
}

} // namespace test
} // namespace vkai
