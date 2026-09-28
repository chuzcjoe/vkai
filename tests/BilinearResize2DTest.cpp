#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "layers/BilinearResize2D.h"
#include "utils/BufferUtils.h"

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

void ExpectResizeMatchesPyTorch(const std::string& name) {
  const auto directory =
      std::filesystem::path(VKAI_SOURCE_DIR) / "python/pspnet/test_data/bilinear_resize";
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

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          vkai::kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, expected.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           vkai::kHostVisibleMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });
  vkai::BilinearResize2D resize(&context, channels, input_height, input_width, output_height,
                                output_width, batch);
  ASSERT_EQ(resize.OutputHeight(), output_height);
  ASSERT_EQ(resize.OutputWidth(), output_width);
  resize.Init();
  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  resize.Execute(command_buffer.buffer(), input_buffer, output_buffer);
  command_buffer.EndOneTimeCommands();
  std::vector<float> actual(expected.size());
  output_buffer.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 2e-5F + 2e-5F * std::abs(expected[i])) << "index " << i;
  }
}

}  // namespace

namespace vkai {
namespace test {

TEST(BilinearResize2DTest, MatchesAllPyramidPoolingUpsamples) {
  ExpectResizeMatchesPyTorch("ppm_scale_1");
  ExpectResizeMatchesPyTorch("ppm_scale_2");
  ExpectResizeMatchesPyTorch("ppm_scale_3");
  ExpectResizeMatchesPyTorch("ppm_scale_6");
}

TEST(BilinearResize2DTest, MatchesFinalLogitsUpsample) {
  ExpectResizeMatchesPyTorch("final_logits");
}

TEST(BilinearResize2DTest, MatchesBatchedNonSquareResize) {
  ExpectResizeMatchesPyTorch("batched_non_square");
}

}  // namespace test
}  // namespace vkai
