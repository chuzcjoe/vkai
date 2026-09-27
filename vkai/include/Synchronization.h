#pragma once

#include <vulkan/vulkan.h>

#include "VulkanBuffer.h"

namespace vkai {

class Synchronization final {
 public:
  Synchronization() = delete;

  static void InsertComputeBarrier(const VkCommandBuffer& command_buffer);

  static void InsertHostReadBarrier(const VkCommandBuffer& command_buffer,
                                    const core::vulkan::VulkanBuffer& buffer);
};

}  // namespace vkai
