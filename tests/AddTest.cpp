#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "Add.h"
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

void ExpectMNIST12AddMatchesReference(const std::filesystem::path& input_path,
                                      const std::filesystem::path& addend_path,
                                      const std::filesystem::path& output_path, int channels,
                                      int elements_per_channel) {
  constexpr VkMemoryPropertyFlags kHostVisibleMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  std::vector<float> input;
  std::vector<float> addend;
  std::vector<float> reference;
  ASSERT_TRUE(ReadFloatBinary(input_path, input));
  ASSERT_TRUE(ReadFloatBinary(addend_path, addend));
  ASSERT_TRUE(ReadFloatBinary(output_path, reference));
  ASSERT_EQ(input.size(), static_cast<size_t>(channels * elements_per_channel));
  ASSERT_EQ(addend.size(), static_cast<size_t>(channels));
  ASSERT_EQ(reference.size(), input.size());

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer addend_buffer(&context, addend.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, reference.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });
  addend_buffer.MapData(
      [&addend](void* data) { std::memcpy(data, addend.data(), addend.size() * sizeof(float)); });

  vkai::Add add(&context, channels, elements_per_channel);
  add.Init();
  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  add.Execute(command_buffer.buffer(), input_buffer, addend_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(reference.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t i = 0; i < reference.size(); ++i) {
    const float tolerance = 1e-6F + 1e-6F * std::abs(reference[i]);
    EXPECT_NEAR(actual[i], reference[i], tolerance) << "Add output mismatch at element " << i;
  }
}

}  // namespace

namespace vkai {
namespace test {

TEST(AddTest, MatchesAllMNIST12AddLayers) {
  const std::filesystem::path data_dir =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/mnist12_cnn/test_data";
  ExpectMNIST12AddMatchesReference(data_dir / "002_Plus30_Add_input_0.bin",
                                   data_dir / "002_Plus30_Add_input_1.bin",
                                   data_dir / "002_Plus30_Add_output.bin", 8, 28 * 28);
  ExpectMNIST12AddMatchesReference(data_dir / "006_Plus112_Add_input_0.bin",
                                   data_dir / "006_Plus112_Add_input_1.bin",
                                   data_dir / "006_Plus112_Add_output.bin", 16, 14 * 14);
  ExpectMNIST12AddMatchesReference(data_dir / "012_Plus214_Add_input_0.bin",
                                   data_dir / "012_Plus214_Add_input_1.bin",
                                   data_dir / "012_Plus214_Add_output.bin", 10, 1);
}

}  // namespace test
}  // namespace vkai
