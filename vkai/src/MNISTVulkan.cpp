#include "MNISTVulkan.h"

#include <cstring>
#include <iostream>

namespace vkai {
namespace {

constexpr int kInputSize = 28 * 28;
constexpr int kFC1OutputSize = 128;
constexpr int kFC2OutputSize = 10;
constexpr int kBatchSize = 1;

constexpr VkMemoryPropertyFlags kHostVisibleMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

} // namespace

MNISTVulkan::MNISTVulkan(core::vulkan::VulkanContext *context,
                         const std::string &weights_file,
                         core::vulkan::VulkanBuffer &input,
                         core::vulkan::VulkanBuffer &output)
    : context_(context), input_buffer_(input), output_buffer_(output) {
  if (!LoadWeights(weights_file)) {
    return;
  }

  if (weights_.fc1_weights.size() != kInputSize * kFC1OutputSize ||
      weights_.fc1_bias.size() != kFC1OutputSize ||
      weights_.fc2_weights.size() != kFC1OutputSize * kFC2OutputSize ||
      weights_.fc2_bias.size() != kFC2OutputSize) {
    std::cerr << "Unexpected MNIST weight dimensions\n";
    return;
  }

  CreateBuffers();
  UploadWeights();

  fc1_layer_ = std::make_unique<Linear>(
      context_, input_buffer_, fc1_weights_buffer_, fc1_bias_buffer_,
      fc1_output_buffer_, kInputSize, kFC1OutputSize, kBatchSize);
  relu1_layer_ = std::make_unique<Relu>(context_, fc1_output_buffer_,
                                        relu1_output_buffer_, kFC1OutputSize);
  fc2_layer_ = std::make_unique<Linear>(
      context_, relu1_output_buffer_, fc2_weights_buffer_, fc2_bias_buffer_,
      fc2_output_buffer_, kFC1OutputSize, kFC2OutputSize, kBatchSize);
  softmax_layer_ = std::make_unique<Softmax>(
      context_, fc2_output_buffer_, output_buffer_, kFC2OutputSize, kBatchSize);
}

bool MNISTVulkan::LoadWeights(const std::string &weights_file) {
  return WeightLoader::Load(weights_file, weights_);
}

void MNISTVulkan::CreateBuffers() {
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
      context_, sizeof(float) * kFC1OutputSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  fc2_weights_buffer_ = core::vulkan::VulkanBuffer(
      context_, weights_.fc2_weights.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  fc2_bias_buffer_ = core::vulkan::VulkanBuffer(
      context_, weights_.fc2_bias.size() * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);

  fc2_output_buffer_ = core::vulkan::VulkanBuffer(
      context_, sizeof(float) * kFC2OutputSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemory);
}

void MNISTVulkan::UploadWeights() {
  fc1_weights_buffer_.MapData([this](void *data) {
    std::memcpy(data, weights_.fc1_weights.data(),
                weights_.fc1_weights.size() * sizeof(float));
  });
  fc1_bias_buffer_.MapData([this](void *data) {
    std::memcpy(data, weights_.fc1_bias.data(),
                weights_.fc1_bias.size() * sizeof(float));
  });
  fc2_weights_buffer_.MapData([this](void *data) {
    std::memcpy(data, weights_.fc2_weights.data(),
                weights_.fc2_weights.size() * sizeof(float));
  });
  fc2_bias_buffer_.MapData([this](void *data) {
    std::memcpy(data, weights_.fc2_bias.data(),
                weights_.fc2_bias.size() * sizeof(float));
  });
}

void MNISTVulkan::Init() {
  fc1_layer_->Init();
  relu1_layer_->Init();
  fc2_layer_->Init();
  softmax_layer_->Init();
}

void MNISTVulkan::Run(const VkCommandBuffer command_buffer) {
  const VkMemoryBarrier compute_barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
  };

  const VkBufferMemoryBarrier host_read_barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = output_buffer_.buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  };

  fc1_layer_->Execute(command_buffer);
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &compute_barrier, 0, nullptr, 0, nullptr);

  relu1_layer_->Execute(command_buffer);
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &compute_barrier, 0, nullptr, 0, nullptr);

  fc2_layer_->Execute(command_buffer);
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &compute_barrier, 0, nullptr, 0, nullptr);

  softmax_layer_->Execute(command_buffer);
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                       &host_read_barrier, 0, nullptr);
}

} // namespace vkai
