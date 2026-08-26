#pragma once

#include <memory>
#include <string>

#include "VulkanBuffer.h"
#include "VulkanCompute.h"

#include "Layer.h"
#include "Linear.h"
#include "Relu.h"
#include "Softmax.h"
#include "WeightLoader.h"

namespace vkai {

class MNISTVulkan {
public:
  explicit MNISTVulkan(core::vulkan::VulkanContext *context,
                       const std::string &weights_file,
                       core::vulkan::VulkanBuffer &input,
                       core::vulkan::VulkanBuffer &output);

  void Init();

  void Run(const VkCommandBuffer &command_buffer);

private:
  bool LoadWeights(const std::string &weights_file);

  void CreateBuffers();

  void UploadWeights();

  void InsertComputeBarrier(const VkCommandBuffer &command_buffer);

  void InsertHostReadBarrier(const VkCommandBuffer &command_buffer);

  core::vulkan::VulkanContext *context_;
  core::vulkan::VulkanBuffer &input_buffer_;
  core::vulkan::VulkanBuffer &output_buffer_;

  core::vulkan::VulkanBuffer fc1_weights_buffer_;
  core::vulkan::VulkanBuffer fc1_bias_buffer_;
  core::vulkan::VulkanBuffer fc1_output_buffer_;
  core::vulkan::VulkanBuffer relu1_output_buffer_;
  core::vulkan::VulkanBuffer fc2_weights_buffer_;
  core::vulkan::VulkanBuffer fc2_bias_buffer_;
  core::vulkan::VulkanBuffer fc2_output_buffer_;

  ModelWeights weights_;

  std::vector<std::unique_ptr<Layer>> layers_;

  //   std::unique_ptr<Linear> fc1_layer_;
  //   std::unique_ptr<Relu> relu1_layer_;
  //   std::unique_ptr<Linear> fc2_layer_;
  //   std::unique_ptr<Softmax> softmax_layer_;
};

} // namespace vkai
