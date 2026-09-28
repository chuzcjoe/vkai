#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "ChannelConcat.h"
#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"

namespace {

bool ReadFloats(const std::filesystem::path& path, std::vector<float>* values) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return false;
  const auto bytes = stream.tellg();
  if (bytes < 0 || bytes % sizeof(float) != 0) return false;
  values->resize(static_cast<size_t>(bytes) / sizeof(float));
  stream.seekg(0);
  return bytes == 0 ||
         static_cast<bool>(stream.read(reinterpret_cast<char*>(values->data()), bytes));
}

void ExpectChannelConcatMatchesPyTorch(const std::string& name) {
  const auto directory =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/pspnet/test_data/channel_concat";
  std::ifstream metadata(directory / (name + ".txt"));
  int batch, height, width, input_count;
  ASSERT_TRUE(metadata >> batch >> height >> width >> input_count);
  std::vector<int> channels(input_count);
  for (int& channel_count : channels) ASSERT_TRUE(metadata >> channel_count);
  std::vector<std::vector<float>> inputs(input_count);
  size_t output_channels = 0;
  for (int index = 0; index < input_count; ++index) {
    ASSERT_TRUE(ReadFloats(directory / (name + "_input_" + std::to_string(index) + ".bin"),
                           &inputs[index]));
    ASSERT_EQ(inputs[index].size(), static_cast<size_t>(batch * channels[index] * height * width));
    output_channels += channels[index];
  }
  std::vector<float> expected;
  ASSERT_TRUE(ReadFloats(directory / (name + "_output.bin"), &expected));
  ASSERT_EQ(expected.size(), static_cast<size_t>(batch * output_channels * height * width));

  constexpr auto kMemory =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  std::vector<std::unique_ptr<core::vulkan::VulkanBuffer>> input_buffers;
  std::vector<const core::vulkan::VulkanBuffer*> input_buffer_pointers;
  for (const auto& input : inputs) {
    input_buffers.push_back(std::make_unique<core::vulkan::VulkanBuffer>(
        &context, input.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kMemory));
    input_buffers.back()->MapData(
        [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });
    input_buffer_pointers.push_back(input_buffers.back().get());
  }
  core::vulkan::VulkanBuffer output_buffer(&context, expected.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kMemory);
  vkai::ChannelConcat concat(&context, channels, height, width, batch);
  ASSERT_EQ(concat.OutputChannels(), static_cast<int>(output_channels));
  concat.Init();
  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  concat.Execute(command_buffer.buffer(), input_buffer_pointers, output_buffer);
  command_buffer.EndOneTimeCommands();
  std::vector<float> actual(expected.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1e-6F + 1e-6F * std::abs(expected[index]))
        << "index " << index;
  }
}

}  // namespace

namespace vkai {
namespace test {

TEST(ChannelConcatTest, MatchesFiveInputPyramidPoolingConcat) {
  ExpectChannelConcatMatchesPyTorch("pspnet_ppm");
}

TEST(ChannelConcatTest, MatchesTwoInputBatchedConcat) {
  ExpectChannelConcatMatchesPyTorch("batched_two_inputs");
}

}  // namespace test
}  // namespace vkai
