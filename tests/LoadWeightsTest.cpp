#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>

#include "WeightLoader.h"

namespace vkai {
namespace test {

TEST(LoadWeightsTest, MNIST) {
  ModelWeights weights;
  ASSERT_TRUE(WeightLoader::Load((std::filesystem::path(VKAI_SOURCE_DIR) /
                                  "python/mnist/mnist_weights.bin")
                                     .string(),
                                 weights));
  std::cout << "fc1_weights size: " << weights.fc1_weights.size() << '\n';
}

} // namespace test
} // namespace vkai
