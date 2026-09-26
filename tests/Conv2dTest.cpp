#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "Conv2D.h"
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

}  // namespace

namespace vkai {
namespace test {

TEST(Conv2dTest, MatchesMNIST12FirstConvolution) {
  constexpr int kInputChannels = 1;
  constexpr int kOutputChannels = 8;
  constexpr int kInputHeight = 28;
  constexpr int kInputWidth = 28;
  constexpr int kKernelHeight = 5;
  constexpr int kKernelWidth = 5;
  constexpr int kStrideHeight = 1;
  constexpr int kStrideWidth = 1;
  constexpr int kPaddingHeight = 2;
  constexpr int kPaddingWidth = 2;
  constexpr int kBatchSize = 1;
  constexpr VkMemoryPropertyFlags kHostVisibleMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  const std::filesystem::path data_dir =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist12_cnn/test_data";
  std::vector<float> input;
  std::vector<float> weights;
  std::vector<float> reference;
  ASSERT_TRUE(ReadFloatBinary(data_dir / "000_input.bin", input));
  ASSERT_TRUE(ReadFloatBinary(data_dir / "001_Convolution28_Conv_input_1.bin", weights));
  ASSERT_TRUE(ReadFloatBinary(data_dir / "001_Convolution28_Conv_output.bin", reference));

  ASSERT_EQ(input.size(),
            static_cast<size_t>(kBatchSize * kInputChannels * kInputHeight * kInputWidth));
  ASSERT_EQ(weights.size(),
            static_cast<size_t>(kOutputChannels * kInputChannels * kKernelHeight * kKernelWidth));
  ASSERT_EQ(reference.size(),
            static_cast<size_t>(kBatchSize * kOutputChannels * kInputHeight * kInputWidth));

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();

  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, reference.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });

  Conv2D conv2d(&context, weights, kInputChannels, kOutputChannels, kInputHeight, kInputWidth,
                kKernelHeight, kKernelWidth, kStrideHeight, kStrideWidth, kPaddingHeight,
                kPaddingWidth, PaddingType::Zero, kBatchSize);
  ASSERT_EQ(conv2d.OutputHeight(), kInputHeight);
  ASSERT_EQ(conv2d.OutputWidth(), kInputWidth);
  conv2d.Init();

  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  conv2d.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });

  for (size_t i = 0; i < reference.size(); ++i) {
    const float tolerance = 1e-5F + 1e-5F * std::abs(reference[i]);
    EXPECT_NEAR(actual[i], reference[i], tolerance) << "Conv2D output mismatch at element " << i;
  }
}

TEST(Conv2dTest, SupportsReflectPadding) {
  constexpr VkMemoryPropertyFlags kHostVisibleMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const std::vector<float> input = {
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F,
  };
  const std::vector<float> weights(9, 1.0F);
  const std::vector<float> reference = {
      33.0F, 36.0F, 39.0F, 42.0F, 45.0F, 48.0F, 51.0F, 54.0F, 57.0F,
  };

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();

  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, reference.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });

  Conv2D conv2d(&context, weights, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, PaddingType::Reflect);
  conv2d.Init();

  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  conv2d.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  EXPECT_EQ(actual, reference);
}

}  // namespace test
}  // namespace vkai
