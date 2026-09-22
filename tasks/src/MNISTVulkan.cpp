#include "MNISTVulkan.h"

#include <iostream>
#include <utility>

namespace vkai {
namespace {

constexpr int kInputSize = 28 * 28;
constexpr int kFC1OutputSize = 128;
constexpr int kFC2OutputSize = 10;
constexpr int kBatchSize = 1;

}  // namespace

MNISTVulkan::MNISTVulkan(core::vulkan::VulkanContext* context, const std::string& weights_file,
                         core::vulkan::VulkanBuffer& input, core::vulkan::VulkanBuffer& output)
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

  if (input_buffer_.Size() < kInputSize * sizeof(float)) {
    std::cerr << "MNIST input buffer is too small\n";
    return;
  }
  if (output_buffer_.Size() < kFC1OutputSize * sizeof(float)) {
    std::cerr << "MNIST output buffer must fit the largest intermediate tensor\n";
    return;
  }

  layers_.emplace_back(std::make_unique<Linear>(context_, weights_.fc1_weights, weights_.fc1_bias,
                                                kInputSize, kFC1OutputSize, kBatchSize));
  layers_.emplace_back(std::make_unique<Relu>(context_, kFC1OutputSize));
  layers_.emplace_back(std::make_unique<Linear>(context_, weights_.fc2_weights, weights_.fc2_bias,
                                                kFC1OutputSize, kFC2OutputSize, kBatchSize));
  layers_.emplace_back(std::make_unique<Softmax>(context_, kFC2OutputSize, kBatchSize));
}

bool MNISTVulkan::LoadWeights(const std::string& weights_file) {
  return MNISTWeightsLoader::Load(weights_file, weights_);
}

void MNISTVulkan::Init() {
  for (auto& layer : layers_) {
    layer->Init();
  }
}

void MNISTVulkan::InsertComputeBarrier(const VkCommandBuffer& command_buffer) {
  const VkMemoryBarrier compute_barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
  };

  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &compute_barrier, 0, nullptr, 0,
                       nullptr);
}

void MNISTVulkan::InsertHostReadBarrier(const VkCommandBuffer& command_buffer,
                                        const core::vulkan::VulkanBuffer& buffer) {
  const VkBufferMemoryBarrier host_read_barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffer.buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  };

  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &host_read_barrier, 0,
                       nullptr);
}

core::vulkan::VulkanBuffer& MNISTVulkan::Run(const VkCommandBuffer& command_buffer) {
  core::vulkan::VulkanBuffer* input_buffer = &input_buffer_;
  core::vulkan::VulkanBuffer* output_buffer = &output_buffer_;

  for (auto& layer : layers_) {
    layer->Execute(command_buffer, *input_buffer, *output_buffer);
    InsertComputeBarrier(command_buffer);
    std::swap(input_buffer, output_buffer);
  }
  InsertHostReadBarrier(command_buffer, *input_buffer);
  return *input_buffer;
}

}  // namespace vkai
