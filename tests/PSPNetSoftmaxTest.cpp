#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "Softmax.h"
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

void ExpectChannelSoftmaxMatchesPyTorch(const std::string& name) {
  const auto directory = std::filesystem::path(VKAI_SOURCE_DIR) / "python/pspnet/test_data/softmax";
  std::ifstream shape_stream(directory / (name + ".txt"));
  int batch, channels, height, width;
  ASSERT_TRUE(shape_stream >> batch >> channels >> height >> width);
  std::vector<float> input;
  std::vector<float> expected;
  ASSERT_TRUE(ReadFloats(directory / (name + "_input.bin"), &input));
  ASSERT_TRUE(ReadFloats(directory / (name + "_output.bin"), &expected));
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

TEST(PSPNetSoftmaxTest, MatchesFinalPSPNetChannelSoftmax) {
  ExpectChannelSoftmaxMatchesPyTorch("pspnet_final");
}

TEST(PSPNetSoftmaxTest, MatchesBatchedExtremeChannelSoftmax) {
  ExpectChannelSoftmaxMatchesPyTorch("batched_extreme");
}

}  // namespace test
}  // namespace vkai
