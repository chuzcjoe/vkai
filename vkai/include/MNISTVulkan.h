#pragma once

#include <iostream>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCompute.h"

#include "WeightLoader.h"

namespace vkai {

class MNISTVulkan {
public:
  explicit MNISTVulkan(core::vulkan::VulkanContext *context,
                       const std::string &weigths_file);

private:
  void LoadWeights(const std::string &weights_file);

  void CreateBuffers();

  void UploadWeights();

  core::vulkan::VulkanContext *context_;

  core::vulkan::VulkanBuffer input_buffer_;
  core::vulkan::VulkanBuffer fc1_weights_buffer_;
  core::vulkan::VulkanBuffer fc1_bias_buffer_;
  core::vulkan::VulkanBuffer fc1_output_buffer_;
  core::vulkan::VulkanBuffer relu1_output_buffer_;
  core::vulkan::VulkanBuffer fc2_weights_buffer_;
  core::vulkan::VulkanBuffer fc2_bias_buffer_;
  core::vulkan::VulkanBuffer fc2_output_buffer_;
  core::vulkan::VulkanBuffer softmax_output_buffer_;

  ModelWeights weights_;
};

} // namespace vkai