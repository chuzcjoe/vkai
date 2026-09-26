#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "MaxPool2D.h"
#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"

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

void ExpectMNIST12PoolMatchesReference(const std::filesystem::path& input_path,
                                       const std::filesystem::path& output_path, int channels,
                                       int input_height, int input_width, int kernel_height,
                                       int kernel_width, int stride_height, int stride_width) {
  constexpr VkMemoryPropertyFlags kHostVisibleMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  std::vector<float> input;
  std::vector<float> reference;
  ASSERT_TRUE(ReadFloatBinary(input_path, input));
  ASSERT_TRUE(ReadFloatBinary(output_path, reference));

  const int output_height = (input_height - kernel_height) / stride_height + 1;
  const int output_width = (input_width - kernel_width) / stride_width + 1;
  ASSERT_EQ(input.size(), static_cast<size_t>(channels * input_height * input_width));
  ASSERT_EQ(reference.size(), static_cast<size_t>(channels * output_height * output_width));

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, reference.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });

  vkai::MaxPool2D max_pool(&context, channels, input_height, input_width, kernel_height,
                           kernel_width, stride_height, stride_width);
  ASSERT_EQ(max_pool.OutputHeight(), output_height);
  ASSERT_EQ(max_pool.OutputWidth(), output_width);
  max_pool.Init();

  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  max_pool.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t i = 0; i < reference.size(); ++i) {
    const float tolerance = 1e-6F + 1e-6F * std::abs(reference[i]);
    EXPECT_NEAR(actual[i], reference[i], tolerance) << "MaxPool2D output mismatch at element " << i;
  }
}

}  // namespace

namespace vkai {
namespace test {

TEST(MaxPool2DTest, MatchesBothMNIST12PoolingLayers) {
  const std::filesystem::path data_dir =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist12_cnn/test_data";

  ExpectMNIST12PoolMatchesReference(data_dir / "004_Pooling66_MaxPool_input_0.bin",
                                    data_dir / "004_Pooling66_MaxPool_output.bin", 8, 28, 28, 2, 2,
                                    2, 2);
  ExpectMNIST12PoolMatchesReference(data_dir / "008_Pooling160_MaxPool_input_0.bin",
                                    data_dir / "008_Pooling160_MaxPool_output.bin", 16, 14, 14, 3,
                                    3, 3, 3);
}

}  // namespace test
}  // namespace vkai
