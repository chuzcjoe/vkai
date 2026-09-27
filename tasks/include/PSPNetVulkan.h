#pragma once

#include <memory>
#include <string>

#include "VulkanBuffer.h"

namespace vkai {

// Runs the pretrained ResNet-50-dilated + PPM PSPNet inference graph on Vulkan.
class PSPNetVulkan {
 public:
  PSPNetVulkan(core::vulkan::VulkanContext* context, const std::string& weights_file,
               int input_height, int input_width, core::vulkan::VulkanBuffer& input,
               core::vulkan::VulkanBuffer& output);
  ~PSPNetVulkan();

  PSPNetVulkan(const PSPNetVulkan&) = delete;
  PSPNetVulkan& operator=(const PSPNetVulkan&) = delete;

  void Init();
  core::vulkan::VulkanBuffer& Run(const VkCommandBuffer& command_buffer);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vkai
