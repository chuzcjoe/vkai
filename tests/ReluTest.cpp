#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "Relu.h"
#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"

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

TEST(ReluTest, MatchesPyTorchReference) {
  constexpr VkMemoryPropertyFlags kHostVisibleMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  const std::filesystem::path data_dir =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/test_data";
  std::vector<float> input;
  std::vector<float> reference;
  ASSERT_TRUE(ReadFloatBinary(data_dir / "fc1_output.bin", input));
  ASSERT_TRUE(ReadFloatBinary(data_dir / "relu1_output.bin", reference));
  ASSERT_EQ(input.size(), reference.size());
  ASSERT_FALSE(input.empty());

  core::vulkan::VulkanContext context(
      false, core::vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  context.Init();

  core::vulkan::VulkanBuffer input_buffer(
      &context, input.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(
      &context, reference.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  input_buffer.MapData([&input](void *data) {
    std::memcpy(data, input.data(), input.size() * sizeof(float));
  });

  Relu relu(&context, input_buffer, output_buffer,
            static_cast<int>(input.size()));
  relu.Init();

  auto command_buffer =
      core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  relu.Execute(command_buffer.buffer());
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  output_buffer.MapData([&actual](void *data) {
    std::memcpy(actual.data(), data, actual.size() * sizeof(float));
  });

  for (size_t i = 0; i < reference.size(); ++i) {
    EXPECT_NEAR(actual[i], reference[i], 1e-6F)
        << "ReLU output mismatch at element " << i;
  }
}

} // namespace test
} // namespace vkai
