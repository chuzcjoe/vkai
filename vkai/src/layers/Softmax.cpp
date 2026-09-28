#include "layers/Softmax.h"

#include <cstring>
#include <filesystem>
#include <iostream>

#include "utils/BufferUtils.h"

namespace vkai {

Softmax::Softmax(core::vulkan::VulkanContext* context, int input_size, int batch_size)
    : Layer(context),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      kHostVisibleMemory),
      uniform_data_{.channel_count = input_size,
                    .batch_size = batch_size,
                    .input_height = 1,
                    .input_width = 1,
                    .mode = static_cast<int>(SoftmaxMode::kContiguousPerBatch),
                    .reserved_0 = 0,
                    .reserved_1 = 0,
                    .reserved_2 = 0} {
  if (input_size <= 0 || batch_size <= 0) {
    std::cerr << "Softmax dimensions must be positive\n";
    return;
  }
  uniform_buffer_.MapData(
      [this](void* data) { std::memcpy(data, &uniform_data_, sizeof(UniformData)); });
  valid_ = true;
}

Softmax::Softmax(core::vulkan::VulkanContext* context, int channel_count, int input_height,
                 int input_width, int batch_size, SoftmaxMode mode)
    : Layer(context),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      kHostVisibleMemory),
      uniform_data_{.channel_count = channel_count,
                    .batch_size = batch_size,
                    .input_height = input_height,
                    .input_width = input_width,
                    .mode = static_cast<int>(mode),
                    .reserved_0 = 0,
                    .reserved_1 = 0,
                    .reserved_2 = 0} {
  if (channel_count <= 0 || input_height <= 0 || input_width <= 0 || batch_size <= 0 ||
      mode != SoftmaxMode::kChannelNCHW) {
    std::cerr << "Softmax NCHW dimensions must be positive and mode must be kChannelNCHW\n";
    return;
  }
  uniform_buffer_.MapData(
      [this](void* data) { std::memcpy(data, &uniform_data_, sizeof(UniformData)); });
  valid_ = true;
}

void Softmax::Init() {
  if (!valid_) {
    std::cerr << "Cannot initialize an invalid Softmax\n";
    return;
  }
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  vkUpdateDescriptorSets(context_->logical_device, 1, &writes_[0], 0, nullptr);

  const std::string cache = GetPipelineCache();
  if (!cache.empty() && !std::filesystem::exists(cache)) {
    SavePipelineCache(cache);
  }
}

void Softmax::Execute(const VkCommandBuffer& command_buffer,
                      const core::vulkan::VulkanBuffer& input_buffer,
                      core::vulkan::VulkanBuffer& output_buffer) {
  if (!valid_ || pipeline == VK_NULL_HANDLE) {
    std::cerr << "Cannot run an invalid or uninitialized Softmax\n";
    return;
  }
  const VkDeviceSize required_size = static_cast<VkDeviceSize>(uniform_data_.batch_size) *
                                     uniform_data_.channel_count * uniform_data_.input_height *
                                     uniform_data_.input_width * sizeof(float);
  if (input_buffer.Size() < required_size || output_buffer.Size() < required_size) {
    std::cerr << "Softmax input or output buffer is too small\n";
    return;
  }
  UpdateStorageBufferDescriptors(1, 2, input_buffer, output_buffer);

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);

  constexpr uint32_t kLocalSize = 64;
  const uint32_t vectors = static_cast<uint32_t>(uniform_data_.batch_size) *
                           (uniform_data_.mode == static_cast<int>(SoftmaxMode::kChannelNCHW)
                                ? uniform_data_.input_height * uniform_data_.input_width
                                : 1);
  const uint32_t group_x = (vectors + kLocalSize - 1) / kLocalSize;
  vkCmdDispatch(command_buffer, group_x, 1, 1);
}

std::vector<core::vulkan::BindingInfo> Softmax::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& Softmax::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "Softmax.comp.spv"
      ;
  return shader_code;
}

const std::string Softmax::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  if (cache_dir.empty()) {
    return "";
  }
  return cache_dir + "/softmax.cache";
}

}  // namespace vkai
