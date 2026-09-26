#pragma once

#include "Layer.h"

namespace vkai {

enum class AddMode {
  // The addend has channel_count elements and is broadcast over NCHW spatial elements and batches.
  kChannelBias,
  // Input and addend have the same NCHW shape and are added element by element.
  kElementwise,
};

// Adds either a per-channel tensor or a same-shape tensor to an NCHW input.
class Add : public Layer {
 public:
  Add(core::vulkan::VulkanContext* context, int channel_count, int elements_per_channel,
      int batch_size = 1, AddMode mode = AddMode::kChannelBias);

  void Init() override;

  // Add requires a second input buffer; use the overload below for execution.
  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               core::vulkan::VulkanBuffer& output_buffer) override;

  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               const core::vulkan::VulkanBuffer& addend_buffer,
               core::vulkan::VulkanBuffer& output_buffer);

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;

  const std::vector<uint32_t>& LoadShaderCode() const override;

  const std::string GetPipelineCache() const override;

 private:
  struct alignas(16) UniformData {
    int channel_count;
    int elements_per_channel;
    int batch_size;
    int element_count;
    int mode;
    int reserved_0;
    int reserved_1;
    int reserved_2;
  } uniform_data_;

  core::vulkan::VulkanBuffer uniform_buffer_;
  bool valid_ = false;
};

}  // namespace vkai
