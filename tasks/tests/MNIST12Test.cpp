#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "MNIST12Vulkan.h"
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

TEST(MNIST12Test, MatchesPyTorchLogits) {
  constexpr VkMemoryPropertyFlags kHostVisibleMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  constexpr size_t kLargestIntermediateElements = 8U * 28U * 28U;
  const std::filesystem::path mnist12_dir =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist12_cnn";
  const std::filesystem::path data_dir = mnist12_dir / "test_data";

  std::vector<float> input;
  std::vector<float> reference;
  ASSERT_TRUE(ReadFloatBinary(data_dir / "000_input.bin", input));
  ASSERT_TRUE(ReadFloatBinary(data_dir / "model_logits.bin", reference));
  ASSERT_EQ(input.size(), 28U * 28U);
  ASSERT_EQ(reference.size(), 10U);

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, kLargestIntermediateElements * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });

  MNIST12Vulkan mnist12(&context, (mnist12_dir / "model/mnist12_weights.bin").string(),
                        input_buffer, output_buffer);
  mnist12.Init();
  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  core::vulkan::VulkanBuffer& result_buffer = mnist12.Run(command_buffer.buffer());
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  result_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t index = 0; index < reference.size(); ++index) {
    const float tolerance = 1e-4F + 1e-4F * std::abs(reference[index]);
    EXPECT_NEAR(actual[index], reference[index], tolerance)
        << "MNIST-12 logit mismatch at class " << index;
  }
}

}  // namespace test
}  // namespace vkai
