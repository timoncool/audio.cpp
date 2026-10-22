#include "engine/framework/assets/transcribe_cpp_gguf.h"

#include "engine/framework/assets/gguf_metadata.h"

#include <stdexcept>
#include <unordered_map>

namespace engine::assets::transcribe_cpp {

NameRules compile_rules(const std::vector<std::pair<const char *, const char *>> & table) {
    NameRules rules;
    rules.reserve(table.size());
    for (const auto & [pattern, replacement] : table) rules.emplace_back(std::regex(pattern), replacement);
    return rules;
}

namespace {

const NameRules & conformer_rules() {
    static const NameRules rules = compile_rules({
        {R"(^enc\.blocks\.(\d+)\.norm_ff([12])\.(weight|bias)$)", "encoder.layers.$1.norm_feed_forward$2.$3"},
        {R"(^enc\.blocks\.(\d+)\.ff([12])\.(linear[12])\.(weight|bias)$)", "encoder.layers.$1.feed_forward$2.$3.$4"},
        {R"(^enc\.blocks\.(\d+)\.norm_attn\.(weight|bias)$)", "encoder.layers.$1.norm_self_att.$2"},
        {R"(^enc\.blocks\.(\d+)\.attn\.linear_q\.(weight|bias)$)", "encoder.layers.$1.self_attn.q_proj.$2"},
        {R"(^enc\.blocks\.(\d+)\.attn\.linear_k\.(weight|bias)$)", "encoder.layers.$1.self_attn.k_proj.$2"},
        {R"(^enc\.blocks\.(\d+)\.attn\.linear_v\.(weight|bias)$)", "encoder.layers.$1.self_attn.v_proj.$2"},
        {R"(^enc\.blocks\.(\d+)\.attn\.linear_out\.(weight|bias)$)", "encoder.layers.$1.self_attn.o_proj.$2"},
        {R"(^enc\.blocks\.(\d+)\.attn\.linear_pos\.weight$)", "encoder.layers.$1.self_attn.relative_k_proj.weight"},
        {R"(^enc\.blocks\.(\d+)\.attn\.pos_bias_([uv])$)", "encoder.layers.$1.self_attn.bias_$2"},
        {R"(^enc\.blocks\.(\d+)\.norm_conv\.(weight|bias)$)", "encoder.layers.$1.norm_conv.$2"},
        {R"(^enc\.blocks\.(\d+)\.conv\.pointwise([12])\.(weight|bias)$)", "encoder.layers.$1.conv.pointwise_conv$2.$3"},
        {R"(^enc\.blocks\.(\d+)\.conv\.depthwise\.(weight|bias)$)", "encoder.layers.$1.conv.depthwise_conv.$2"},
        {R"(^enc\.blocks\.(\d+)\.conv\.bn\.(weight|bias|running_mean|running_var)$)", "encoder.layers.$1.conv.norm.$2"},
        {R"(^enc\.blocks\.(\d+)\.norm_out\.(weight|bias)$)", "encoder.layers.$1.norm_out.$2"},
        {R"(^pred\.embed\.weight$)", "decoder.embedding.weight"},
        {R"(^pred\.lstm\.(\d+)\.Wx$)", "decoder.lstm.weight_ih_l$1"},
        {R"(^pred\.lstm\.(\d+)\.Wh$)", "decoder.lstm.weight_hh_l$1"},
        {R"(^pred\.lstm\.(\d+)\.bias$)", "decoder.lstm.bias_ih_l$1"},
        {R"(^joint\.pred\.(weight|bias)$)", "decoder.decoder_projector.$1"},
        {R"(^joint\.enc\.(weight|bias)$)", "encoder_projector.$1"},
        {R"(^joint\.out\.(weight|bias)$)", "joint.head.$1"},
    });
    return rules;
}

}  // namespace

std::optional<std::string> rename(const std::string & name, const NameRules & model_rules) {
    for (const auto * rules : {&model_rules, &conformer_rules()}) {
        for (const auto & [pattern, replacement] : *rules) {
            if (std::regex_match(name, pattern)) return std::regex_replace(name, pattern, replacement);
        }
    }
    return std::nullopt;
}

bool is_file(const std::filesystem::path & path, const char * head_kind) {
    if (!std::filesystem::is_regular_file(path) || path.extension() != ".gguf") return false;
    const GgufMetadata metadata(path);
    return metadata.has("stt.parakeet.head_kind") && metadata.require_string("stt.parakeet.head_kind") == head_kind;
}

std::shared_ptr<const TensorSource> open_renamed(const std::filesystem::path & path, const NameRules & model_rules) {
    auto source = open_tensor_source(path);
    std::unordered_map<std::string, std::string> routes;
    for (const auto & tensor : source->tensors()) {
        const auto hf_name = rename(tensor.name, model_rules);
        if (!hf_name) throw std::runtime_error("unsupported tensor in transcribe.cpp GGUF " + path.string() + ": " + tensor.name);
        routes.emplace(*hf_name, tensor.name);
    }
    return make_renamed_tensor_source(std::move(source), std::move(routes));
}

}  // namespace engine::assets::transcribe_cpp
