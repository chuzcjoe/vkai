#pragma once

#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanCompute.h"

namespace vkai {

class Layer : public core::vulkan::VulkanCompute {
 public:
  explicit Layer(core::vulkan::VulkanContext* context) : core::vulkan::VulkanCompute(context) {}
  virtual ~Layer() = default;

  virtual void Execute(const VkCommandBuffer& command_buffer) = 0;
};

}  // namespace vkai