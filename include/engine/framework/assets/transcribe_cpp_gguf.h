#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

// transcribe.cpp writes NeMo FastConformer checkpoints (Parakeet, Nemotron) as one GGUF in NeMo tensor naming, with
// the model config under stt.* keys and the vocabulary under tokenizer.ggml.*. These helpers let a model load such a
// file in place of its own package: the conformer blocks, predictor and joint share one naming, each model adds the
// names of its own subsampling and extra heads.
namespace engine::assets::transcribe_cpp {

using NameRules = std::vector<std::pair<std::regex, std::string>>;

NameRules compile_rules(const std::vector<std::pair<const char *, const char *>> & table);

// The Hugging Face name of a tensor of the file: the model's own rules first, then the shared conformer naming.
std::optional<std::string> rename(const std::string & name, const NameRules & model_rules);

// A transcribe.cpp GGUF of this NeMo head kind (tdt, rnnt).
bool is_file(const std::filesystem::path & path, const char * head_kind);

// The file's tensors under their Hugging Face names; a tensor no rule names is an error, not skipped.
std::shared_ptr<const TensorSource> open_renamed(const std::filesystem::path & path, const NameRules & model_rules);

}  // namespace engine::assets::transcribe_cpp
