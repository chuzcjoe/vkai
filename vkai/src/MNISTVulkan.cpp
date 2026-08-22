#include "MNISTVulkan.h"

namespace vkai {

MNISTVulkan::MNISTVulkan(core::vulkan::VulkanContext *context,
                         const std::string &weights_file)
    : context_(context) {
  LoadWeights(weights_file);
  CreateBuffers();
}

MNISTVulkan::CreateBuffers() {
  input_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * 28 * 28, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  fc1_weights_buffer_ = core::vulkan::VulkanBuffer(
      context_, weights_.fc1_weights.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  fc1_bias_buffer_ = core::vulkan::VulkanBuffer(
      context_, weights_.fc1_bias.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  fc1_output_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  relu1_output_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  fc2_weights_buffer_ = core::vulkan::VulkanBuffer(
      context_, weights_.fc2_weights.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  fc2_bias_buffer_ = core::vulkan::VulkanBuffer(
      context_, weights_.fc2_bias.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  fc2_output_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  softmax_output_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void MNISTVulkan::LoadWeights() {
  fc1_weights_buffer.MapData([this](void *data) {
    std::memcpy(data, weights_.fc1_weights.data(),
                weights_.fc1_weights.size() * sizeof(float));
  });

  fc1_bias_buffer.MapData([this](void *data) {
    std::memcpy(data, weights_.fc1_bias.data(),
                weights_.fc1_bias.size() * sizeof(float));
  });

  fc2_weights_buffer.MapData([this](void *data) {
    std::memcpy(data, weights_.fc2_weights.data(),
                weights_.fc2_weights.size() * sizeof(float));
  });

  fc2_bias_buffer.MapData([this](void *data) {
    std::memcpy(data, weights_.fc2_bias.data(),
                weights_.fc2_bias.size() * sizeof(float));
  });
}

} // namespace vkai