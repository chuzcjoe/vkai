#include "LinearLayer.h"

#include <filesystem>

namespace vkai {

LinearLayer::LinearLayer(core::vulkan::VulkanContext *context,
                         core::vulkan::VulkanBuffer &input,
                         core::vulkan::VulkanBuffer &weights,
                         core::vulkan::VulkanBuffer &bias,
                         core::vulkan::VulkanBuffer &output)
    : VulkanCompute(context), input_buffer_(input), weights_buffer_(weights),
      bias_buffer_(bias), output_buffer_(output),
      uniform_buffer_(context, sizeof(UniformData),
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      uniform_data_{.input_size = 100, .output_size = 100, .batch_size = 1} {

  // Copy uniform data to the uniform buffer
  uniform_buffer_.MapData([this](void *data) {
    memcpy(data, &uniform_data_, sizeof(UniformData));
  });
}

void LinearLayer::Init() {
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  CreateStorageBufferDescriptorSet(1, input_buffer_);
  CreateStorageBufferDescriptorSet(2, weights_buffer_);
  CreateStorageBufferDescriptorSet(3, bias_buffer_);
  CreateStorageBufferDescriptorSet(4, output_buffer_);

  vkUpdateDescriptorSets(context_->logical_device, writes_.size(),
                         writes_.data(), 0, nullptr);

  // Save cache if file doesn't exist
  if (!std::filesystem::exists(GetPipelineCache())) {
    SavePipelineCache(GetPipelineCache());
  }
}

void LinearLayer::Run(const VkCommandBuffer command_buffer) {
  // Record commands to dispatch the compute shader
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline_layout, 0, 1, &descriptor_set_, 0, nullptr);

  const uint32_t group_x = (uniform_data_.output_size + 255) / 256;
  vkCmdDispatch(command_buffer, group_x, 1, 1);
}

std::vector<core::vulkan::BindingInfo> LinearLayer::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t> &LinearLayer::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "LinearLayer.comp.spv"
      ;
  return shader_code;
}

const std::string LinearLayer::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  if (cache_dir.empty()) {
    return "";
  }
  std::string pipeline_cache = cache_dir + "/linear_layer.cache";
  printf("Using pipeline cache file: %s\n", pipeline_cache.c_str());
  return pipeline_cache;
}

} // namespace vkai