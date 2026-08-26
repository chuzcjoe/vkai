#pragma once

#include <memory>
#include <string>

#include "Layer.h"
#include "Linear.h"
#include "Relu.h"
#include "Softmax.h"
#include "VulkanBuffer.h"
#include "VulkanCompute.h"
#include "WeightLoader.h"

namespace vkai {

class MNISTVulkan {
 public:
  explicit MNISTVulkan(core::vulkan::VulkanContext* context, const std::string& weights_file,
                       core::vulkan::VulkanBuffer& input, core::vulkan::VulkanBuffer& output);

  void Init();

  core::vulkan::VulkanBuffer& Run(const VkCommandBuffer& command_buffer);

 private:
  bool LoadWeights(const std::string& weights_file);

  void InsertComputeBarrier(const VkCommandBuffer& command_buffer);

  void InsertHostReadBarrier(const VkCommandBuffer& command_buffer,
                             const core::vulkan::VulkanBuffer& buffer);

  core::vulkan::VulkanContext* context_;
  core::vulkan::VulkanBuffer& input_buffer_;
  core::vulkan::VulkanBuffer& output_buffer_;

  ModelWeights weights_;

  std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace vkai
