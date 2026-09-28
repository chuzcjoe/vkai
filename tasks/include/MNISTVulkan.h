#pragma once

#include <memory>
#include <string>

#include "VulkanBuffer.h"
#include "VulkanCompute.h"
#include "layers/Layer.h"
#include "layers/Linear.h"
#include "layers/Relu.h"
#include "layers/Softmax.h"
#include "utils/WeightsLoader.h"

namespace vkai {

class MNISTVulkan {
 public:
  MNISTVulkan(core::vulkan::VulkanContext* context, const std::string& weights_file,
              core::vulkan::VulkanBuffer& input, core::vulkan::VulkanBuffer& output);

  void Init();

  core::vulkan::VulkanBuffer& Run(const VkCommandBuffer& command_buffer);

 private:
  bool LoadWeights(const std::string& weights_file);

  core::vulkan::VulkanContext* context_;
  core::vulkan::VulkanBuffer& input_buffer_;
  core::vulkan::VulkanBuffer& output_buffer_;

  Weights weights_;

  std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace vkai
