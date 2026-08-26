#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "MNISTVulkan.h"
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

TEST(MNISTTest, MatchesPyTorchReference) {
  constexpr VkMemoryPropertyFlags kHostVisibleMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  const std::filesystem::path source_dir = VKAI_SOURCE_DIR;
  const std::filesystem::path mnist_dir = source_dir / "python/mnist";
  const std::filesystem::path data_dir = mnist_dir / "test_data";

  std::vector<float> input;
  std::vector<float> reference;
  ASSERT_TRUE(ReadFloatBinary(data_dir / "input.bin", input));
  ASSERT_TRUE(ReadFloatBinary(data_dir / "softmax_output.bin", reference));
  ASSERT_EQ(input.size(), 28U * 28U);
  ASSERT_EQ(reference.size(), 10U);

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

  MNISTVulkan mnist(&context, (mnist_dir / "mnist_weights.bin").string(),
                    input_buffer, output_buffer);
  mnist.Init();

  auto command_buffer =
      core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  mnist.Run(command_buffer.buffer());
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  output_buffer.MapData([&actual](void *data) {
    std::memcpy(actual.data(), data, actual.size() * sizeof(float));
  });
  ASSERT_EQ(actual.size(), reference.size());

  for (size_t i = 0; i < reference.size(); ++i) {
    EXPECT_NEAR(actual[i], reference[i], 1e-5F)
        << "MNIST output mismatch at class " << i;
  }
  EXPECT_NEAR(std::accumulate(actual.begin(), actual.end(), 0.0F), 1.0F, 1e-5F);
}

} // namespace test
} // namespace vkai
