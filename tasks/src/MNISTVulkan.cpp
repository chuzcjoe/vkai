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

  const WeightTensor* fc1_weights = weights_.Find("fc1.weight");
  const WeightTensor* fc1_bias = weights_.Find("fc1.bias");
  const WeightTensor* fc2_weights = weights_.Find("fc2.weight");
  const WeightTensor* fc2_bias = weights_.Find("fc2.bias");
  if (fc1_weights == nullptr || fc1_bias == nullptr || fc2_weights == nullptr ||
      fc2_bias == nullptr ||
      fc1_weights->data.size() != static_cast<size_t>(kInputSize * kFC1OutputSize) ||
      fc1_bias->data.size() != static_cast<size_t>(kFC1OutputSize) ||
      fc2_weights->data.size() != static_cast<size_t>(kFC1OutputSize * kFC2OutputSize) ||
      fc2_bias->data.size() != static_cast<size_t>(kFC2OutputSize)) {
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

  layers_.emplace_back(std::make_unique<Linear>(context_, fc1_weights->data, fc1_bias->data,
                                                kInputSize, kFC1OutputSize, kBatchSize));
  layers_.emplace_back(std::make_unique<Relu>(context_, kFC1OutputSize));
  layers_.emplace_back(std::make_unique<Linear>(context_, fc2_weights->data, fc2_bias->data,
                                                kFC1OutputSize, kFC2OutputSize, kBatchSize));
  layers_.emplace_back(std::make_unique<Softmax>(context_, kFC2OutputSize, kBatchSize));
}

bool MNISTVulkan::LoadWeights(const std::string& weights_file) {
  std::string error_message;
  if (!WeightsLoader::Load(weights_file, weights_, &error_message)) {
    std::cerr << error_message << '\n';
    return false;
  }
  return true;
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
