#include "MNIST12Vulkan.h"

#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

namespace vkai {
namespace {

constexpr int kBatchSize = 1;
constexpr int kInputChannels = 1;
constexpr int kInputHeight = 28;
constexpr int kInputWidth = 28;
constexpr int kConv1Channels = 8;
constexpr int kConv2Channels = 16;
constexpr int kKernelSize = 5;
constexpr int kConvPadding = 2;
constexpr int kPool1Size = 2;
constexpr int kPool2Size = 3;
constexpr int kPool1Height = 14;
constexpr int kPool1Width = 14;
constexpr int kPool2Height = 4;
constexpr int kPool2Width = 4;
constexpr int kOutputClasses = 10;
constexpr VkMemoryPropertyFlags kHostVisibleMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

bool HasShape(const WeightTensor* tensor, std::initializer_list<uint32_t> expected_shape) {
  const std::vector<uint32_t> expected(expected_shape);
  return tensor != nullptr && tensor->shape == expected &&
         tensor->data.size() == std::accumulate(expected_shape.begin(), expected_shape.end(),
                                                size_t{1}, std::multiplies<>());
}

}  // namespace

MNIST12Vulkan::MNIST12Vulkan(core::vulkan::VulkanContext* context, const std::string& weights_file,
                             core::vulkan::VulkanBuffer& input, core::vulkan::VulkanBuffer& output)
    : context_(context), input_buffer_(input), output_buffer_(output) {
  if (!LoadWeights(weights_file) || !InitializeLayers()) {
    return;
  }

  constexpr size_t kInputElements = kInputChannels * kInputHeight * kInputWidth;
  constexpr size_t kLargestIntermediateElements = kConv1Channels * kInputHeight * kInputWidth;
  if (input_buffer_.Size() < kInputElements * sizeof(float)) {
    std::cerr << "MNIST-12 input buffer is too small\n";
    return;
  }
  if (output_buffer_.Size() < kLargestIntermediateElements * sizeof(float)) {
    std::cerr << "MNIST-12 output buffer must fit the largest intermediate tensor\n";
    return;
  }
  workspace_buffer_ = std::make_unique<core::vulkan::VulkanBuffer>(
      context_, kLargestIntermediateElements * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      kHostVisibleMemory);
  valid_ = true;
}

bool MNIST12Vulkan::LoadWeights(const std::string& weights_file) {
  std::string error_message;
  if (!WeightsLoader::Load(weights_file, weights_, &error_message)) {
    std::cerr << error_message << '\n';
    return false;
  }
  return true;
}

bool MNIST12Vulkan::InitializeLayers() {
  const WeightTensor* conv1_weight = weights_.Find("conv1.weight");
  const WeightTensor* conv1_bias = weights_.Find("conv1.bias");
  const WeightTensor* conv2_weight = weights_.Find("conv2.weight");
  const WeightTensor* conv2_bias = weights_.Find("conv2.bias");
  const WeightTensor* fc_weight = weights_.Find("fc.weight");
  const WeightTensor* fc_bias = weights_.Find("fc.bias");
  if (!HasShape(conv1_weight, {kConv1Channels, kInputChannels, kKernelSize, kKernelSize}) ||
      !HasShape(conv1_bias, {kConv1Channels}) ||
      !HasShape(conv2_weight, {kConv2Channels, kConv1Channels, kKernelSize, kKernelSize}) ||
      !HasShape(conv2_bias, {kConv2Channels}) ||
      !HasShape(fc_weight, {kOutputClasses, kConv2Channels * kPool2Height * kPool2Width}) ||
      !HasShape(fc_bias, {kOutputClasses})) {
    std::cerr << "Unexpected MNIST-12 weight dimensions\n";
    return false;
  }

  conv1_ = std::make_unique<Conv2D>(context_, conv1_weight->data, kInputChannels, kConv1Channels,
                                    kInputHeight, kInputWidth, kKernelSize, kKernelSize, 1, 1,
                                    kConvPadding, kConvPadding, PaddingType::Zero, kBatchSize);
  add1_ = std::make_unique<Add>(context_, kConv1Channels, kInputHeight * kInputWidth, kBatchSize);
  relu1_ = std::make_unique<Relu>(context_, kConv1Channels * kInputHeight * kInputWidth);
  pool1_ = std::make_unique<MaxPool2D>(context_, kConv1Channels, kInputHeight, kInputWidth,
                                       kPool1Size, kPool1Size, kPool1Size, kPool1Size, kBatchSize);
  conv2_ = std::make_unique<Conv2D>(context_, conv2_weight->data, kConv1Channels, kConv2Channels,
                                    kPool1Height, kPool1Width, kKernelSize, kKernelSize, 1, 1,
                                    kConvPadding, kConvPadding, PaddingType::Zero, kBatchSize);
  add2_ = std::make_unique<Add>(context_, kConv2Channels, kPool1Height * kPool1Width, kBatchSize);
  relu2_ = std::make_unique<Relu>(context_, kConv2Channels * kPool1Height * kPool1Width);
  pool2_ = std::make_unique<MaxPool2D>(context_, kConv2Channels, kPool1Height, kPool1Width,
                                       kPool2Size, kPool2Size, kPool2Size, kPool2Size, kBatchSize);
  fc_ = std::make_unique<Linear>(context_, fc_weight->data, fc_bias->data,
                                 kConv2Channels * kPool2Height * kPool2Width, kOutputClasses,
                                 kBatchSize);

  conv1_bias_buffer_ = std::make_unique<core::vulkan::VulkanBuffer>(
      context_, conv1_bias->data.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      kHostVisibleMemory);
  conv2_bias_buffer_ = std::make_unique<core::vulkan::VulkanBuffer>(
      context_, conv2_bias->data.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      kHostVisibleMemory);
  conv1_bias_buffer_->MapData([&conv1_bias](void* data) {
    std::memcpy(data, conv1_bias->data.data(), conv1_bias->data.size() * sizeof(float));
  });
  conv2_bias_buffer_->MapData([&conv2_bias](void* data) {
    std::memcpy(data, conv2_bias->data.data(), conv2_bias->data.size() * sizeof(float));
  });
  return true;
}

void MNIST12Vulkan::Init() {
  if (!valid_) {
    std::cerr << "Cannot initialize an invalid MNIST-12 task\n";
    return;
  }
  conv1_->Init();
  add1_->Init();
  relu1_->Init();
  pool1_->Init();
  conv2_->Init();
  add2_->Init();
  relu2_->Init();
  pool2_->Init();
  fc_->Init();
}

void MNIST12Vulkan::InsertComputeBarrier(const VkCommandBuffer& command_buffer) {
  const VkMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                       nullptr);
}

void MNIST12Vulkan::InsertHostReadBarrier(const VkCommandBuffer& command_buffer,
                                          const core::vulkan::VulkanBuffer& buffer) {
  const VkBufferMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
                                      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                      .buffer = buffer.buffer,
                                      .offset = 0,
                                      .size = VK_WHOLE_SIZE};
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

core::vulkan::VulkanBuffer& MNIST12Vulkan::Run(const VkCommandBuffer& command_buffer) {
  if (!valid_) {
    std::cerr << "Cannot run an invalid MNIST-12 task\n";
    return output_buffer_;
  }
  const auto run_layer = [&](auto& layer, const core::vulkan::VulkanBuffer& input,
                             core::vulkan::VulkanBuffer& output) {
    layer->Execute(command_buffer, input, output);
    InsertComputeBarrier(command_buffer);
  };

  run_layer(conv1_, input_buffer_, output_buffer_);
  add1_->Execute(command_buffer, output_buffer_, *conv1_bias_buffer_, *workspace_buffer_);
  InsertComputeBarrier(command_buffer);
  run_layer(relu1_, *workspace_buffer_, output_buffer_);
  run_layer(pool1_, output_buffer_, *workspace_buffer_);
  run_layer(conv2_, *workspace_buffer_, output_buffer_);
  add2_->Execute(command_buffer, output_buffer_, *conv2_bias_buffer_, *workspace_buffer_);
  InsertComputeBarrier(command_buffer);
  run_layer(relu2_, *workspace_buffer_, output_buffer_);
  run_layer(pool2_, output_buffer_, *workspace_buffer_);
  run_layer(fc_, *workspace_buffer_, output_buffer_);
  InsertHostReadBarrier(command_buffer, output_buffer_);
  return output_buffer_;
}

}  // namespace vkai
