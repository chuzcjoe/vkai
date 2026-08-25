#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "LinearLayer.h"
#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "WeightLoader.h"

namespace {

bool ReadFloatBinary(const std::filesystem::path &path,
                     std::vector<float> &values) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Failed to open reference data: " << path << '\n';
    return false;
  }

  const std::streamsize byte_size = file.tellg();
  if (byte_size < 0 || byte_size % sizeof(float) != 0) {
    std::cerr << "Invalid float32 reference data: " << path << '\n';
    return false;
  }

  values.resize(static_cast<size_t>(byte_size) / sizeof(float));
  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char *>(values.data()), byte_size)) {
    std::cerr << "Failed to read reference data: " << path << '\n';
    values.clear();
    return false;
  }
  return true;
}

} // namespace

namespace vkai {
namespace test {

TEST(LinearLayerTest, test) {
  constexpr int kInputSize = 28 * 28;
  constexpr int kOutputSize = 128;
  constexpr int kBatchSize = 1;
  constexpr VkMemoryPropertyFlags kHostVisibleMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  const std::filesystem::path source_dir = VKAI_SOURCE_DIR;
  std::vector<float> input;
  std::vector<float> reference;
  ModelWeights weights;
  ASSERT_TRUE(
      ReadFloatBinary(source_dir / "python/test_data/input.bin", input));
  ASSERT_TRUE(ReadFloatBinary(source_dir / "python/test_data/fc1_output.bin",
                              reference));
  ASSERT_TRUE(WeightLoader::Load(
      (source_dir / "python/mnist_weights.bin").string(), weights));

  ASSERT_EQ(input.size(), static_cast<size_t>(kInputSize * kBatchSize));
  ASSERT_EQ(weights.fc1_weights.size(),
            static_cast<size_t>(kInputSize * kOutputSize));
  ASSERT_EQ(weights.fc1_bias.size(), static_cast<size_t>(kOutputSize));
  ASSERT_EQ(reference.size(), static_cast<size_t>(kOutputSize * kBatchSize));

  core::vulkan::VulkanContext context(
      false, core::vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();

  core::vulkan::VulkanBuffer input_buffer(
      &context, input.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer weights_buffer(
      &context, weights.fc1_weights.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer bias_buffer(
      &context, weights.fc1_bias.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(
      &context, reference.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  input_buffer.MapData([&input](void *data) {
    std::memcpy(data, input.data(), input.size() * sizeof(float));
  });
  weights_buffer.MapData([&weights](void *data) {
    std::memcpy(data, weights.fc1_weights.data(),
                weights.fc1_weights.size() * sizeof(float));
  });
  bias_buffer.MapData([&weights](void *data) {
    std::memcpy(data, weights.fc1_bias.data(),
                weights.fc1_bias.size() * sizeof(float));
  });

  LinearLayer fc1_layer(&context, input_buffer, weights_buffer, bias_buffer,
                        output_buffer, kInputSize, kOutputSize, kBatchSize);
  fc1_layer.Init();

  // GPU encode and run
  auto command_buffer =
      core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  fc1_layer.Run(command_buffer.buffer());
  command_buffer.EndOneTimeCommands();

  // Read back gpu results
  std::vector<float> actual(reference.size());
  output_buffer.MapData([&actual](void *data) {
    std::memcpy(actual.data(), data, actual.size() * sizeof(float));
  });
  ASSERT_EQ(actual.size(), reference.size());

  for (size_t i = 0; i < reference.size(); ++i) {
    const float tolerance = 1e-4F + 1e-5F * std::abs(reference[i]);
    EXPECT_NEAR(actual[i], reference[i], tolerance)
        << "FC1 output mismatch at element " << i;
  }
}

} // namespace test
} // namespace vkai
