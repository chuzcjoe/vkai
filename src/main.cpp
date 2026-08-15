#include <iostream>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include "model_generated.h"

namespace {

bool RunFlatBuffersCase() {
  flatbuffers::FlatBufferBuilder builder;
  const auto name = builder.CreateString("Alice");
  const auto email = builder.CreateString("alice@example.com");
  const auto person = vkai::fbs::CreatePerson(builder, name, 7, email);
  vkai::fbs::FinishPersonBuffer(builder, person);

  flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
  if (!vkai::fbs::VerifyPersonBuffer(verifier)) {
    std::cerr << "FlatBuffers verification failed\n";
    return false;
  }

  const auto* decoded = vkai::fbs::GetPerson(builder.GetBufferPointer());
  std::cout << "FlatBuffers: " << decoded->name()->str() << ", id=" << decoded->id()
            << ", bytes=" << builder.GetSize() << '\n';
  return decoded->id() == 7 && decoded->email()->str() == "alice@example.com";
}

}  // namespace

int main() {
  const bool ok = RunFlatBuffersCase();
  std::cout << (ok ? "FlatBuffers case passed.\n" : "FlatBuffers case failed.\n");
  return ok ? 0 : 1;
}
