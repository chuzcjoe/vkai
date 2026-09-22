#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>

#include "MNISTWeightsLoader.h"

namespace vkai {
namespace test {

TEST(LoadWeightsTest, MNIST) {
  MNISTWeights weights;
  ASSERT_TRUE(MNISTWeightsLoader::Load(
      (std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist/mnist_weights.bin").string(),
      weights));
  std::cout << "fc1_weights size: " << weights.fc1_weights.size() << '\n';
}

}  // namespace test
}  // namespace vkai
