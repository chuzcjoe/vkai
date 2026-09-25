#include "WeightsLoader.h"

#include <fstream>
#include <limits>
#include <utility>
#include <vector>

#include "model_generated.h"

namespace vkai {
namespace {

bool SetError(Weights& weights, std::string* error_message, const std::string& message) {
  weights = {};
  if (error_message != nullptr) {
    *error_message = message;
  }
  return false;
}

}  // namespace

const WeightTensor* Weights::Find(std::string_view tensor_name) const {
  const auto iterator = tensors_.find(std::string(tensor_name));
  return iterator == tensors_.end() ? nullptr : &iterator->second;
}

bool Weights::Contains(std::string_view tensor_name) const { return Find(tensor_name) != nullptr; }

bool WeightsLoader::Load(const std::string& filename, Weights& weights,
                         std::string* error_message) {
  weights = {};
  if (error_message != nullptr) {
    error_message->clear();
  }

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    return SetError(weights, error_message, "Failed to open weights file: " + filename);
  }

  file.seekg(0, std::ios::end);
  const std::streamoff file_size = file.tellg();
  if (file_size <= 0) {
    return SetError(weights, error_message, "Weights file is empty: " + filename);
  }
  if (static_cast<uintmax_t>(file_size) > std::numeric_limits<size_t>::max()) {
    return SetError(weights, error_message, "Weights file is too large: " + filename);
  }
  file.seekg(0, std::ios::beg);

  const size_t buffer_size = static_cast<size_t>(file_size);
  std::vector<uint8_t> buffer(buffer_size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
    return SetError(weights, error_message, "Failed to read weights file: " + filename);
  }

  flatbuffers::Verifier verifier(buffer.data(), buffer.size());
  if (!fbs::VerifyModelBuffer(verifier)) {
    return SetError(weights, error_message, "Invalid FlatBuffers weights file: " + filename);
  }

  const fbs::Model* model = fbs::GetModel(buffer.data());
  if (model->name() == nullptr || model->name()->str().empty()) {
    return SetError(weights, error_message, "Weights model name is empty: " + filename);
  }

  Weights loaded_weights;
  loaded_weights.name_ = model->name()->str();
  const auto* tensors = model->weights();
  if (tensors == nullptr) {
    return SetError(weights, error_message, "Weights tensor list is missing: " + filename);
  }

  for (flatbuffers::uoffset_t index = 0; index < tensors->size(); ++index) {
    const fbs::Tensor* tensor = tensors->Get(index);
    if (tensor->name() == nullptr || tensor->name()->str().empty() || tensor->shape() == nullptr ||
        tensor->data() == nullptr) {
      return SetError(weights, error_message,
                      "Weights tensor is missing required fields: " + filename);
    }

    size_t element_count = 1;
    for (uint32_t dimension : *tensor->shape()) {
      if (dimension != 0U && element_count > std::numeric_limits<size_t>::max() / dimension) {
        return SetError(weights, error_message, "Weights tensor shape overflows: " + filename);
      }
      element_count *= dimension;
    }
    if (element_count != tensor->data()->size()) {
      return SetError(weights, error_message,
                      "Weights tensor shape does not match data length: " + filename);
    }

    WeightTensor loaded_tensor{
        .shape = std::vector<uint32_t>(tensor->shape()->begin(), tensor->shape()->end()),
        .data = std::vector<float>(tensor->data()->begin(), tensor->data()->end()),
    };
    const std::string tensor_name = tensor->name()->str();
    if (!loaded_weights.tensors_.emplace(tensor_name, std::move(loaded_tensor)).second) {
      return SetError(weights, error_message, "Duplicate weights tensor name: " + tensor_name);
    }
  }

  weights = std::move(loaded_weights);
  return true;
}

}  // namespace vkai
