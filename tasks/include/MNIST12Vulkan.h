#pragma once

#include <memory>
#include <string>

#include "Add.h"
#include "Conv2D.h"
#include "Linear.h"
#include "MaxPool2D.h"
#include "Relu.h"
#include "VulkanBuffer.h"
#include "WeightsLoader.h"

namespace vkai {

// Executes the ONNX Model Zoo MNIST-12 graph using Vulkan compute layers.
class MNIST12Vulkan {
 public:
  MNIST12Vulkan(core::vulkan::VulkanContext* context, const std::string& weights_file,
                core::vulkan::VulkanBuffer& input, core::vulkan::VulkanBuffer& output);

  void Init();

  core::vulkan::VulkanBuffer& Run(const VkCommandBuffer& command_buffer);

 private:
  bool LoadWeights(const std::string& weights_file);
  bool InitializeLayers();
  void InsertComputeBarrier(const VkCommandBuffer& command_buffer);
  void InsertHostReadBarrier(const VkCommandBuffer& command_buffer,
                             const core::vulkan::VulkanBuffer& buffer);

  core::vulkan::VulkanContext* context_;
  core::vulkan::VulkanBuffer& input_buffer_;
  core::vulkan::VulkanBuffer& output_buffer_;
  Weights weights_;

  std::unique_ptr<Conv2D> conv1_;
  std::unique_ptr<Add> add1_;
  std::unique_ptr<Relu> relu1_;
  std::unique_ptr<MaxPool2D> pool1_;
  std::unique_ptr<Conv2D> conv2_;
  std::unique_ptr<Add> add2_;
  std::unique_ptr<Relu> relu2_;
  std::unique_ptr<MaxPool2D> pool2_;
  std::unique_ptr<Linear> fc_;
  std::unique_ptr<core::vulkan::VulkanBuffer> conv1_bias_buffer_;
  std::unique_ptr<core::vulkan::VulkanBuffer> conv2_bias_buffer_;
  std::unique_ptr<core::vulkan::VulkanBuffer> workspace_buffer_;
  bool valid_ = false;
};

}  // namespace vkai
