#include <fstream>
#include <iostream>

#include "WeightLoader.h"

namespace vkai {
namespace {

bool ReadExact(std::ifstream& file, void* data, std::streamsize size) {
  return static_cast<bool>(file.read(reinterpret_cast<char*>(data), size));
}

}  // namespace

bool WeightLoader::Load(const std::string& filename, ModelWeights& weights) {
  weights = {};

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open weights file: " << filename << '\n';
    return false;
  }

  // Read and verify magic number
  uint32_t magic = 0;
  if (!ReadExact(file, &magic, sizeof(magic))) {
    std::cerr << "Failed to read weights file header: " << filename << '\n';
    return false;
  }
  if (magic != 0x4D4E5354) {  // 'MNST'
    std::cerr << "Invalid weights file format: " << filename << '\n';
    return false;
  }

  // Read version
  uint32_t version = 0;
  if (!ReadExact(file, &version, sizeof(version))) {
    std::cerr << "Failed to read weights file version: " << filename << '\n';
    return false;
  }
  if (version != 1) {
    std::cerr << "Unsupported weights file version " << version << ": " << filename << '\n';
    return false;
  }

  ModelWeights loaded_weights;

  // Helper to read tensor
  auto readTensor = [&file, &filename](std::vector<float>& data) {
    uint32_t count = 0;
    if (!ReadExact(file, &count, sizeof(count))) {
      std::cerr << "Failed to read tensor size: " << filename << '\n';
      return false;
    }

    data.resize(count);
    const auto byte_size = static_cast<std::streamsize>(count * sizeof(float));
    if (!ReadExact(file, data.data(), byte_size)) {
      std::cerr << "Failed to read tensor data: " << filename << '\n';
      return false;
    }

    return true;
  };

  // Read weights in same order as exported
  if (!readTensor(loaded_weights.conv1_weights) || !readTensor(loaded_weights.conv1_bias) ||
      !readTensor(loaded_weights.conv2_weights) || !readTensor(loaded_weights.conv2_bias) ||
      !readTensor(loaded_weights.fc1_weights) || !readTensor(loaded_weights.fc1_bias) ||
      !readTensor(loaded_weights.fc2_weights) || !readTensor(loaded_weights.fc2_bias) ||
      !readTensor(loaded_weights.fc3_weights) || !readTensor(loaded_weights.fc3_bias)) {
    return false;
  }

  weights = std::move(loaded_weights);
  return true;
}

}  // namespace vkai
