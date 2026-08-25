#pragma once

#include <memory>
#include <string>
#include <vector>

#include "VulkanBuffer.h"
#include "VulkanCompute.h"

#include "LinearLayer.h"
#include "WeightLoader.h"

namespace vkai {

class MNISTVulkan {
public:
  explicit MNISTVulkan(core::vulkan::VulkanContext *context,
                       const std::string &weigths_file);

  void Init();

  void UploadInput(const std::vector<float> &input);

  void Run(const VkCommandBuffer command_buffer);

  std::vector<float> DownloadFC1Output();

  bool IsValid() const { return valid_; }

private:
  bool LoadWeights(const std::string &weights_file);

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

  std::unique_ptr<LinearLayer> fc1_layer_;

  bool valid_ = false;
};

} // namespace vkai
