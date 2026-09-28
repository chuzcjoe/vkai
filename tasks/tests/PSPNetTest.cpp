#include <gtest/gtest.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "PSPNetVulkan.h"
#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
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

std::array<unsigned char, 3> PaletteColor(int label) {
  int value = label;
  int red = 0;
  int green = 0;
  int blue = 0;
  for (int bit = 0; bit < 8; ++bit) {
    red |= ((value >> 0) & 1) << (7 - bit);
    green |= ((value >> 1) & 1) << (7 - bit);
    blue |= ((value >> 2) & 1) << (7 - bit);
    value >>= 3;
  }
  return {static_cast<unsigned char>(red), static_cast<unsigned char>(green),
          static_cast<unsigned char>(blue)};
}

unsigned char ToByte(float value) {
  return static_cast<unsigned char>(std::clamp(value, 0.0F, 255.0F));
}

void WriteVulkanVisualization(const std::filesystem::path& output_dir,
                              const std::vector<float>& input,
                              const std::vector<float>& probabilities, int channels, int height,
                              int width) {
  std::vector<unsigned char> mask(static_cast<size_t>(height) * width * 3);
  std::vector<unsigned char> overlay(mask.size());
  constexpr float kMean[] = {0.485F, 0.456F, 0.406F};
  constexpr float kStd[] = {0.229F, 0.224F, 0.225F};
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int label = 0;
      float maximum = probabilities[static_cast<size_t>(y) * width + x];
      for (int channel = 1; channel < channels; ++channel) {
        const float probability =
            probabilities[(static_cast<size_t>(channel) * height + y) * width + x];
        if (probability > maximum) {
          maximum = probability;
          label = channel;
        }
      }
      const auto color = PaletteColor(label);
      const size_t pixel = (static_cast<size_t>(y) * width + x) * 3;
      for (int channel = 0; channel < 3; ++channel) {
        mask[pixel + channel] = color[channel];
        const float source =
            (input[(static_cast<size_t>(channel) * height + y) * width + x] * kStd[channel] +
             kMean[channel]) *
            255.0F;
        overlay[pixel + channel] = ToByte((source + color[channel]) * 0.5F);
      }
    }
  }
  std::filesystem::create_directories(output_dir);
  ASSERT_NE(stbi_write_png((output_dir / "vulkan_mask.png").c_str(), width, height, 3, mask.data(),
                           width * 3),
            0);
  ASSERT_NE(stbi_write_png((output_dir / "vulkan_overlay.png").c_str(), width, height, 3,
                           overlay.data(), width * 3),
            0);
}

}  // namespace

namespace vkai {
namespace test {

TEST(PSPNetTest, MatchesPyTorchProbabilities) {
  const auto pspnet_dir = std::filesystem::path(VKAI_SOURCE_DIR) / "python/pspnet";
  const auto data_dir = pspnet_dir / "test_data/end_to_end";
  std::ifstream shape_stream(data_dir / "shape.txt");
  int batch, channels, height, width;
  ASSERT_TRUE(shape_stream >> batch >> channels >> height >> width);
  ASSERT_EQ(batch, 1);
  ASSERT_EQ(channels, 150);
  std::vector<float> input;
  std::vector<float> expected;
  ASSERT_TRUE(ReadFloats(data_dir / "input.bin", &input));
  ASSERT_TRUE(ReadFloats(data_dir / "probabilities.bin", &expected));
  ASSERT_EQ(input.size(), static_cast<size_t>(3 * height * width));
  ASSERT_EQ(expected.size(), static_cast<size_t>(channels * height * width));

  core::vulkan::VulkanContext context(false, core::vulkan::QueueFamilyType::Compute,
                                      VK_NULL_HANDLE);
  context.Init();
  core::vulkan::VulkanBuffer input_buffer(&context, input.size() * sizeof(float),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  core::vulkan::VulkanBuffer output_buffer(&context, expected.size() * sizeof(float),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
  input_buffer.MapData(
      [&input](void* data) { std::memcpy(data, input.data(), input.size() * sizeof(float)); });
  PSPNetVulkan pspnet(&context, (pspnet_dir / "model/pspnet_weights.bin").string(), height, width,
                      input_buffer, output_buffer);
  pspnet.Init();
  auto command_buffer = core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
  core::vulkan::VulkanBuffer& result = pspnet.Run(command_buffer.buffer());
  command_buffer.EndOneTimeCommands();
  std::vector<float> actual(expected.size());
  result.MapData(
      [&actual](void* data) { std::memcpy(actual.data(), data, actual.size() * sizeof(float)); });
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1e-4F + 1e-4F * std::abs(expected[index]))
        << "probability index " << index;
  }
  const auto output_path = pspnet_dir / "output/vulkan_probabilities.bin";
  std::filesystem::create_directories(output_path.parent_path());
  std::ofstream output_stream(output_path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(output_stream);
  output_stream.write(reinterpret_cast<const char*>(actual.data()),
                      static_cast<std::streamsize>(actual.size() * sizeof(float)));
  ASSERT_TRUE(output_stream);
  WriteVulkanVisualization(pspnet_dir / "output", input, actual, channels, height, width);
}

}  // namespace test
}  // namespace vkai
