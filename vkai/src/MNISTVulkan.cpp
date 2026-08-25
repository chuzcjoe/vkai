#include "MNISTVulkan.h"

#include <cstring>
#include <iostream>

namespace vkai {
namespace {

constexpr int kInputSize = 28 * 28;
constexpr int kFC1OutputSize = 128;
constexpr int kBatchSize = 1;

constexpr VkMemoryPropertyFlags kHostVisibleMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

bool WriteHostVisibleBuffer(const std::vector<float> &values,
                            core::vulkan::VulkanBuffer &buffer) {
  const VkDeviceSize size = values.size() * sizeof(float);
  if (size != buffer.Size()) {
    std::cerr << "Upload size does not match Vulkan buffer size\n";
    return false;
  }

  buffer.MapData(
      [&values, size](void *data) { std::memcpy(data, values.data(), size); });
  return true;
}

std::vector<float> ReadHostVisibleBuffer(core::vulkan::VulkanBuffer &buffer) {
  if (buffer.Size() % sizeof(float) != 0) {
    std::cerr << "Vulkan buffer size is not a multiple of float\n";
    return {};
  }

  std::vector<float> values(buffer.Size() / sizeof(float));
  buffer.MapData([&values](void *data) {
    std::memcpy(values.data(), data, values.size() * sizeof(float));
  });
  return values;
}

} // namespace

MNISTVulkan::MNISTVulkan(core::vulkan::VulkanContext *context,
                         const std::string &weights_file)
    : context_(context) {
  if (!LoadWeights(weights_file)) {
    return;
  }

  if (weights_.fc1_weights.size() != kInputSize * kFC1OutputSize ||
      weights_.fc1_bias.size() != kFC1OutputSize) {
    std::cerr << "Unexpected FC1 weight dimensions\n";
    return;
  }

  CreateBuffers();
  UploadWeights();
  fc1_layer_ = std::make_unique<LinearLayer>(
      context_, input_buffer_, fc1_weights_buffer_, fc1_bias_buffer_,
      fc1_output_buffer_, kInputSize, kFC1OutputSize, kBatchSize);
  valid_ = fc1_layer_->IsValid();
}

bool MNISTVulkan::LoadWeights(const std::string &weights_file) {
  return WeightLoader::Load(weights_file, weights_);
}

void MNISTVulkan::CreateBuffers() {
  input_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * kInputSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      kHostVisibleMemory);

  fc1_weights_buffer_ = core::vulkan::VulkanBuffer(
      context_, weights_.fc1_weights.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  fc1_bias_buffer_ = core::vulkan::VulkanBuffer(
      context_, weights_.fc1_bias.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  fc1_output_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * kFC1OutputSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  relu1_output_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      kHostVisibleMemory);

  fc2_weights_buffer_ = core::vulkan::VulkanBuffer(
      context_, weights_.fc2_weights.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  fc2_bias_buffer_ = core::vulkan::VulkanBuffer(
      context_, weights_.fc2_bias.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  fc2_output_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      kHostVisibleMemory);

  softmax_output_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      kHostVisibleMemory);
}

void MNISTVulkan::UploadWeights() {
  if (!WriteHostVisibleBuffer(weights_.fc1_weights, fc1_weights_buffer_) ||
      !WriteHostVisibleBuffer(weights_.fc1_bias, fc1_bias_buffer_) ||
      !WriteHostVisibleBuffer(weights_.fc2_weights, fc2_weights_buffer_) ||
      !WriteHostVisibleBuffer(weights_.fc2_bias, fc2_bias_buffer_)) {
    valid_ = false;
  }
}

void MNISTVulkan::Init() {
  if (!valid_ || !fc1_layer_) {
    std::cerr << "Cannot initialize an invalid MNISTVulkan model\n";
    return;
  }
  fc1_layer_->Init();
}

void MNISTVulkan::UploadInput(const std::vector<float> &input) {
  if (!valid_) {
    std::cerr << "Cannot upload input to an invalid MNISTVulkan model\n";
    return;
  }
  if (input.size() != kInputSize * kBatchSize) {
    std::cerr << "MNIST input must contain exactly 784 floats\n";
    return;
  }
  if (!WriteHostVisibleBuffer(input, input_buffer_)) {
    valid_ = false;
  }
}

void MNISTVulkan::Run(const VkCommandBuffer command_buffer) {
  if (!valid_ || !fc1_layer_) {
    std::cerr << "Cannot run an invalid MNISTVulkan model\n";
    return;
  }
  fc1_layer_->Run(command_buffer);
}

std::vector<float> MNISTVulkan::DownloadFC1Output() {
  if (!valid_) {
    std::cerr << "Cannot read output from an invalid MNISTVulkan model\n";
    return {};
  }
  return ReadHostVisibleBuffer(fc1_output_buffer_);
}

} // namespace vkai
