#include "Relu.h"

#include <cstring>
#include <filesystem>

namespace vkai {

Relu::Relu(core::vulkan::VulkanContext *context,
           core::vulkan::VulkanBuffer &input,
           core::vulkan::VulkanBuffer &output, int element_count)
    : Layer(context), input_buffer_(input), output_buffer_(output),
      uniform_buffer_(context, sizeof(UniformData),
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      uniform_data_{.element_count = element_count} {
  uniform_buffer_.MapData([this](void *data) {
    std::memcpy(data, &uniform_data_, sizeof(UniformData));
  });
}

void Relu::Init() {
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  CreateStorageBufferDescriptorSet(1, input_buffer_);
  CreateStorageBufferDescriptorSet(2, output_buffer_);
  vkUpdateDescriptorSets(context_->logical_device,
                         static_cast<uint32_t>(writes_.size()), writes_.data(),
                         0, nullptr);

  const std::string cache = GetPipelineCache();
  if (!cache.empty() && !std::filesystem::exists(cache)) {
    SavePipelineCache(cache);
  }
}

void Relu::Execute(const VkCommandBuffer &command_buffer) {
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline_layout, 0, 1, &descriptor_set_, 0, nullptr);

  constexpr uint32_t kLocalSize = 256;
  const uint32_t group_x =
      (static_cast<uint32_t>(uniform_data_.element_count) + kLocalSize - 1) /
      kLocalSize;
  vkCmdDispatch(command_buffer, group_x, 1, 1);
}

std::vector<core::vulkan::BindingInfo> Relu::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t> &Relu::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "Relu.comp.spv"
      ;
  return shader_code;
}

const std::string Relu::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  if (cache_dir.empty()) {
    return "";
  }
  return cache_dir + "/relu.cache";
}

} // namespace vkai
