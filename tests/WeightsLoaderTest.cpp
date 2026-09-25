#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "WeightsLoader.h"
#include "model_generated.h"

namespace {

struct TensorInfo {
  std::string name;
  std::vector<uint32_t> shape;
  std::vector<float> data;
};

std::filesystem::path TestFilePath(const std::string& suffix) {
  const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("vkai_weights_loader_" + suffix + "_" + std::to_string(timestamp) + ".bin");
}

void WriteModel(const std::filesystem::path& path, const std::string& model_name,
                const std::vector<TensorInfo>& tensors) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<vkai::fbs::Tensor>> tensor_offsets;
  tensor_offsets.reserve(tensors.size());
  for (const TensorInfo& tensor : tensors) {
    const auto name = builder.CreateString(tensor.name);
    const auto shape = builder.CreateVector(tensor.shape);
    const auto data = builder.CreateVector(tensor.data);
    tensor_offsets.push_back(vkai::fbs::CreateTensor(builder, name, shape, data));
  }

  const auto name = builder.CreateString(model_name);
  const auto weights = builder.CreateVector(tensor_offsets);
  const auto model = vkai::fbs::CreateModel(builder, name, weights);
  builder.Finish(model, vkai::fbs::ModelIdentifier());

  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             static_cast<std::streamsize>(builder.GetSize()));
  ASSERT_TRUE(file.good());
}

}  // namespace

namespace vkai {
namespace test {

TEST(WeightsLoaderTest, LoadsGeneratedFlatBuffersWeights) {
  Weights weights;
  std::string error_message;
  ASSERT_TRUE(WeightsLoader::Load((std::filesystem::path(VKAI_BINARY_DIR) / "weights.bin").string(),
                                  weights, &error_message))
      << error_message;
  EXPECT_EQ(weights.name(), "tiny_linear");
  ASSERT_TRUE(weights.Contains("linear.weight"));
  ASSERT_TRUE(weights.Contains("linear.bias"));
  const WeightTensor* weight = weights.Find("linear.weight");
  ASSERT_NE(weight, nullptr);
  EXPECT_EQ(weight->shape, (std::vector<uint32_t>{2U, 3U}));
  EXPECT_EQ(weight->data, (std::vector<float>{0.5F, -1.0F, 2.0F, -0.5F, 0.25F, 1.5F}));
}

TEST(WeightsLoaderTest, RejectsMissingAndInvalidFiles) {
  Weights weights;
  std::string error_message;
  ASSERT_TRUE(WeightsLoader::Load((std::filesystem::path(VKAI_BINARY_DIR) / "weights.bin").string(),
                                  weights));
  ASSERT_FALSE(weights.name().empty());

  const std::filesystem::path missing_path = TestFilePath("missing");
  EXPECT_FALSE(WeightsLoader::Load(missing_path.string(), weights, &error_message));
  EXPECT_FALSE(error_message.empty());
  EXPECT_TRUE(weights.name().empty());

  const std::filesystem::path invalid_path = TestFilePath("invalid");
  {
    std::ofstream file(invalid_path, std::ios::binary);
    file << "not a FlatBuffers file";
  }
  EXPECT_FALSE(WeightsLoader::Load(invalid_path.string(), weights, &error_message));
  EXPECT_FALSE(error_message.empty());
  std::filesystem::remove(invalid_path);
}

TEST(WeightsLoaderTest, RejectsInvalidTensorMetadata) {
  const std::filesystem::path duplicate_path = TestFilePath("duplicate");
  WriteModel(duplicate_path, "test", {{"weight", {1U}, {1.0F}}, {"weight", {1U}, {2.0F}}});
  Weights weights;
  EXPECT_FALSE(WeightsLoader::Load(duplicate_path.string(), weights));
  std::filesystem::remove(duplicate_path);

  const std::filesystem::path mismatch_path = TestFilePath("mismatch");
  WriteModel(mismatch_path, "test", {{"weight", {2U}, {1.0F}}});
  EXPECT_FALSE(WeightsLoader::Load(mismatch_path.string(), weights));
  std::filesystem::remove(mismatch_path);

  const std::filesystem::path empty_name_path = TestFilePath("empty_name");
  WriteModel(empty_name_path, "", {{"weight", {1U}, {1.0F}}});
  EXPECT_FALSE(WeightsLoader::Load(empty_name_path.string(), weights));
  std::filesystem::remove(empty_name_path);

  const std::filesystem::path empty_tensor_name_path = TestFilePath("empty_tensor_name");
  WriteModel(empty_tensor_name_path, "test", {{"", {1U}, {1.0F}}});
  EXPECT_FALSE(WeightsLoader::Load(empty_tensor_name_path.string(), weights));
  std::filesystem::remove(empty_tensor_name_path);
}

}  // namespace test
}  // namespace vkai
