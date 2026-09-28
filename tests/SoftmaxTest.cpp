#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "layers/Softmax.h"

namespace {

bool ReadFloatBinary(const std::filesystem::path& path, std::vector<float>& values) {
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
  if (!file.read(reinterpret_cast<char*>(values.data()), byte_size)) {
    std::cerr << "Failed to read reference data: " << path << '\n';
    values.clear();
    return false;
  }
  return true;
}

void ExpectChannelSoftmaxMatchesReference(const std::string& name) {
  const auto directory = std::filesystem::path(VKAI_SOURCE_DIR) / "python/pspnet/test_data/softmax";
  std::ifstream shape_stream(directory / (name + ".txt"));
  int batch, channels, height, width;
  ASSERT_TRUE(shape_stream >> batch >> channels >> height >> width);
  std::vector<float> input;
  std::vector<float> expected;
  ASSERT_TRUE(ReadFloatBinary(directory / (name + "_input.bin"), input));
  ASSERT_TRUE(ReadFloatBinary(directory / (name + "_output.bin"), expected));
  ASSERT_EQ(input.size(), static_cast<size_t>(batch * channels * height * width));
  ASSERT_EQ(expected.size(), input.size());

  constexpr auto kMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, expected.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });
  vkai::Softmax softmax(&context, channels, height, width, batch, vkai::SoftmaxMode::kChannelNCHW);
  softmax.Init();
  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  softmax.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();
  std::vector<float> actual(expected.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 2e-6F + 2e-6F * std::abs(expected[index]))
        << "index " << index;
  }
  for (int n = 0; n < batch; ++n) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float probability_sum = 0.0F;
        for (int c = 0; c < channels; ++c) {
          const size_t index = ((n * channels + c) * height + y) * width + x;
          probability_sum += actual[index];
        }
        EXPECT_NEAR(probability_sum, 1.0F, 2e-6F);
      }
    }
  }
}

}  // namespace

namespace vkai {
namespace test {

TEST(SoftmaxTest, MatchesPyTorchReference) {
  constexpr int kInputSize = 10;
  constexpr int kBatchSize = 1;
  constexpr VkMemoryPropertyFlags kHostVisibleMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  const std::filesystem::path data_dir =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist/test_data";
  std::vector<float> input;
  std::vector<float> reference;
  ASSERT_TRUE(ReadFloatBinary(data_dir / "fc2_output.bin", input));
  ASSERT_TRUE(ReadFloatBinary(data_dir / "softmax_output.bin", reference));
  ASSERT_EQ(input.size(), static_cast<size_t>(kInputSize * kBatchSize));
  ASSERT_EQ(reference.size(), input.size());

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();

  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, reference.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });

  Softmax softmax(&context, kInputSize, kBatchSize);
  softmax.Init();

  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  softmax.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });

  for (size_t i = 0; i < reference.size(); ++i) {
    EXPECT_NEAR(actual[i], reference[i], 1e-6F) << "Softmax output mismatch at element " << i;
  }
  EXPECT_NEAR(std::accumulate(actual.begin(), actual.end(), 0.0F), 1.0F, 1e-6F);
}

TEST(SoftmaxTest, MatchesChannelSoftmaxReference) {
  ExpectChannelSoftmaxMatchesReference("pspnet_final");
}

TEST(SoftmaxTest, MatchesBatchedExtremeChannelSoftmax) {
  ExpectChannelSoftmaxMatchesReference("batched_extreme");
}

}  // namespace test
}  // namespace vkai
