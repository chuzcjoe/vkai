#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "AdaptiveAvgPool2D.h"
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

void ExpectAdaptiveAverageMatchesPyTorch(const std::string& name) {
  const auto directory =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/pspnet/test_data/adaptive_avg_pool";
  std::ifstream shape_stream(directory / (name + ".txt"));
  int batch, channels, input_height, input_width, output_height, output_width;
  ASSERT_TRUE(shape_stream >> batch >> channels >> input_height >> input_width >> output_height >>
              output_width);
  std::vector<float> input;
  std::vector<float> expected;
  ASSERT_TRUE(ReadFloats(directory / (name + "_input.bin"), &input));
  ASSERT_TRUE(ReadFloats(directory / (name + "_output.bin"), &expected));
  ASSERT_EQ(input.size(), static_cast<size_t>(batch * channels * input_height * input_width));
  ASSERT_EQ(expected.size(), static_cast<size_t>(batch * channels * output_height * output_width));

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
  vkai::AdaptiveAvgPool2D pool(&context, channels, input_height, input_width, output_height,
                               output_width, batch);
  ASSERT_EQ(pool.OutputHeight(), output_height);
  ASSERT_EQ(pool.OutputWidth(), output_width);
  pool.Init();
  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  pool.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();

  std::vector<float> actual(expected.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-6F + 1e-6F * std::abs(expected[i])) << "index " << i;
  }
}

}  // namespace

namespace vkai {
namespace test {

TEST(AdaptiveAvgPool2DTest, MatchesAllPyramidPoolingScales) {
  ExpectAdaptiveAverageMatchesPyTorch("pspnet_scale_1");
  ExpectAdaptiveAverageMatchesPyTorch("pspnet_scale_2");
  ExpectAdaptiveAverageMatchesPyTorch("pspnet_scale_3");
  ExpectAdaptiveAverageMatchesPyTorch("pspnet_scale_6");
}

TEST(AdaptiveAvgPool2DTest, MatchesBatchedNonDivisibleRegions) {
  ExpectAdaptiveAverageMatchesPyTorch("batched");
}

}  // namespace test
}  // namespace vkai
