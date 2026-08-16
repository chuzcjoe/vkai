#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <flatbuffers/verifier.h>

#include "model_generated.h"

namespace {

class MappedFile {
 public:
  MappedFile() = default;

  ~MappedFile() {
    if (data_ != MAP_FAILED) {
      munmap(data_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  bool Open(const char* path) {
    fd_ = open(path, O_RDONLY);
    if (fd_ < 0) {
      std::cerr << "open failed: " << std::strerror(errno) << '\n';
      return false;
    }

    struct stat file_stat {};
    if (fstat(fd_, &file_stat) != 0 || file_stat.st_size <= 0) {
      std::cerr << "fstat failed or the weight file is empty\n";
      return false;
    }

    size_ = static_cast<std::size_t>(file_stat.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
      std::cerr << "mmap failed: " << std::strerror(errno) << '\n';
      return false;
    }
    return true;
  }

  const std::uint8_t* data() const {
    return static_cast<const std::uint8_t*>(data_);
  }

  std::size_t size() const { return size_; }

 private:
  int fd_ = -1;
  void* data_ = MAP_FAILED;
  std::size_t size_ = 0;
};

const vkai::fbs::Tensor* FindTensor(const vkai::fbs::Model& model,
                                    const char* tensor_name) {
  if (model.weights() == nullptr) {
    return nullptr;
  }

  for (const auto* tensor : *model.weights()) {
    if (tensor != nullptr && tensor->name() != nullptr &&
        std::strcmp(tensor->name()->c_str(), tensor_name) == 0) {
      return tensor;
    }
  }
  return nullptr;
}

bool RunInference(const vkai::fbs::Model& model) {
  const auto* weight = FindTensor(model, "linear.weight");
  const auto* bias = FindTensor(model, "linear.bias");
  if (weight == nullptr || bias == nullptr || weight->shape() == nullptr ||
      weight->data() == nullptr || bias->data() == nullptr ||
      weight->shape()->size() != 2 || weight->shape()->Get(0) != 2 ||
      weight->shape()->Get(1) != 3 || weight->data()->size() != 6 ||
      bias->data()->size() != 2) {
    std::cerr << "unexpected tensor metadata\n";
    return false;
  }

  constexpr float input[] = {1.0F, 2.0F, 3.0F};
  float output[2] = {};
  for (std::size_t row = 0; row < 2; ++row) {
    output[row] = bias->data()->Get(row);
    for (std::size_t column = 0; column < 3; ++column) {
      output[row] += weight->data()->Get(row * 3 + column) * input[column];
    }
  }

  std::cout << "input:  [1, 2, 3]\n";
  std::cout << "output: [" << output[0] << ", " << output[1] << "]\n";
  return std::fabs(output[0] - 4.6F) < 1e-5F &&
         std::fabs(output[1] - 4.3F) < 1e-5F;
}

}  // namespace

int main(int argc, char** argv) {
  const char* weight_path = argc > 1 ? argv[1] : VKAI_WEIGHT_FILE;

  MappedFile mapped_file;
  if (!mapped_file.Open(weight_path)) {
    return 1;
  }

  flatbuffers::Verifier verifier(mapped_file.data(), mapped_file.size());
  if (!vkai::fbs::VerifyModelBuffer(verifier)) {
    std::cerr << "invalid FlatBuffers weight file\n";
    return 1;
  }

  const auto* model = vkai::fbs::GetModel(mapped_file.data());
  if (model->name() == nullptr) {
    std::cerr << "model name is missing\n";
    return 1;
  }
  std::cout << "mapped " << weight_path << " (" << mapped_file.size() << " bytes)\n";
  std::cout << "model:  " << model->name()->c_str() << '\n';

  if (!RunInference(*model)) {
    std::cerr << "inference failed\n";
    return 1;
  }

  std::cout << "inference passed\n";
  return 0;
}
