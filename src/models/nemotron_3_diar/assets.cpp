#include "engine/models/nemotron_3_diar/assets.h"
#include "engine/models/nemotron_3_diar/frontend.h"

#include "engine/framework/assets/onnx_model.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <cstring>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::nemotron_3_diar {
namespace {

ModelConfig parse_model_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto & encoder = root.require("encoder");
    const auto & head = root.require("head");
    const auto & streaming = root.require("streaming");
    ModelConfig config;
    config.num_speakers = root.require("num_speakers").as_i64();
    config.high_resolution = root.require("high_resolution").as_bool();
    config.output_subsampling_factor = root.require("output_subsampling_factor").as_i64();
    config.encoder.feature_size = encoder.require("feat_in").as_i64();
    config.encoder.hidden_size = encoder.require("hidden_size").as_i64();
    config.encoder.intermediate_size = encoder.require("intermediate_size").as_i64();
    config.encoder.heads = encoder.require("num_attention_heads").as_i64();
    config.encoder.layers = encoder.require("num_hidden_layers").as_i64();
    config.encoder.subsampling_factor = encoder.require("subsampling_factor").as_i64();
    config.encoder.rope_theta = encoder.require("rope_theta").as_f32();
    config.encoder.layer_norm_eps = encoder.require("layer_norm_eps").as_f32();
    config.head.hidden_size = head.require("hidden_size").as_i64();
    config.head.upsample_factor = head.require("upsample_factor").as_i64();
    config.head.upsample_kernel_size = head.require("upsample_kernel_size").as_i64();
    config.streaming.spkcache_len = streaming.require("spkcache_len").as_i64();
    config.streaming.fifo_len = streaming.require("fifo_len").as_i64();
    config.streaming.chunk_len = streaming.require("chunk_len").as_i64();
    config.streaming.chunk_left_context = streaming.require("chunk_left_context").as_i64();
    config.streaming.chunk_right_context = streaming.require("chunk_right_context").as_i64();
    config.streaming.spkcache_update_period = streaming.require("spkcache_update_period").as_i64();
    config.streaming.spkcache_sil_frames_per_spk = streaming.require("spkcache_sil_frames_per_spk").as_i64();
    config.streaming.pred_score_threshold = streaming.require("pred_score_threshold").as_f32();
    config.streaming.scores_boost_latest = streaming.require("scores_boost_latest").as_f32();
    config.streaming.sil_threshold = streaming.require("sil_threshold").as_f32();
    config.streaming.strong_boost_rate = streaming.require("strong_boost_rate").as_f32();
    config.streaming.weak_boost_rate = streaming.require("weak_boost_rate").as_f32();
    config.streaming.min_pos_scores_rate = streaming.require("min_pos_scores_rate").as_f32();
    config.streaming.max_index = streaming.require("max_index").as_i64();
    if (config.num_speakers != 8 || !config.high_resolution ||
        config.encoder.subsampling_factor != config.head.upsample_factor) {
        throw std::runtime_error("Nemotron 3 Diarization configuration is unsupported");
    }
    return config;
}

FeatureConfig parse_feature_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("processor");
    const auto & feature = root.require("feature_extractor");
    FeatureConfig config;
    config.sample_rate = feature.require("sampling_rate").as_i64();
    config.n_fft = feature.require("n_fft").as_i64();
    config.win_length = feature.require("win_length").as_i64();
    config.hop_length = feature.require("hop_length").as_i64();
    config.num_mel_bins = feature.require("feature_size").as_i64();
    config.preemphasis = feature.require("preemphasis").as_f32();
    config.dither = feature.require("dither").as_f32();
    return config;
}

// The parakeet-rs export (nemotron3_diar_v3.onnx) keeps the checkpoint tensors as initializers, except the
// linear layers, which it lowers to MatMul against the [in, out] weight named only by its node. It has no
// mel filterbank (NeMo builds it with librosa and stores it in bf16) and keeps the silence embedding in
// its metadata.
const std::vector<std::pair<std::regex, std::string>> & onnx_matmul_rules() {
    static const std::vector<std::pair<std::regex, std::string>> rules = {
        {std::regex(R"(^/encoder/layers\.(\d+)/attn/(w_qkv|out_proj)/MatMul$)"), "encoder.layers.$1.attn.$2.weight"},
        {std::regex(R"(^/encoder/layers\.(\d+)/ffn/net/net\.(0|3)/MatMul$)"), "encoder.layers.$1.ffn.net.$2.weight"},
        {std::regex(R"(^/pre_encode/proj/MatMul$)"), "encoder.pre_encode.proj.weight"},
        {std::regex(R"(^/(encoder_proj|first_hidden_to_hidden|single_hidden_to_spks)/MatMul$)"),
         "sortformer_modules.$1.weight"},
    };
    return rules;
}

int64_t onnx_metadata_int(const assets::onnx::Model & model, const char * key) {
    const auto it = model.metadata.find(key);
    if (it == model.metadata.end()) {
        throw std::runtime_error("Nemotron 3 Diarization ONNX has no metadata " + std::string(key));
    }
    return std::stoll(it->second);
}

float to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000U;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

std::shared_ptr<const Assets> load_onnx_assets(const std::filesystem::path & path) {
    auto model = std::make_shared<const assets::onnx::Model>(assets::onnx::read_model(path));
    auto result = std::make_shared<Assets>();
    result->resources = assets::ResourceBundle(path.parent_path());
    auto & config = result->model_config;
    const auto & features = result->feature_config;
    if (onnx_metadata_int(*model, "n_spk") != config.num_speakers ||
        onnx_metadata_int(*model, "emb_dim") != config.encoder.hidden_size ||
        onnx_metadata_int(*model, "feat_dim") != features.num_mel_bins ||
        onnx_metadata_int(*model, "subsampling_factor") != config.encoder.subsampling_factor ||
        onnx_metadata_int(*model, "upsample_factor") != config.head.upsample_factor ||
        onnx_metadata_int(*model, "high_resolution") != 1) {
        throw std::runtime_error("Nemotron 3 Diarization ONNX configuration is unsupported: " + path.string());
    }
    config.streaming.spkcache_len = onnx_metadata_int(*model, "spkcache_len");
    config.streaming.fifo_len = onnx_metadata_int(*model, "fifo_len");
    config.streaming.chunk_len = onnx_metadata_int(*model, "chunk_len");
    config.streaming.chunk_right_context = onnx_metadata_int(*model, "right_context");
    config.streaming.spkcache_update_period = onnx_metadata_int(*model, "spkcache_update_period");
    config.streaming.spkcache_sil_frames_per_spk = onnx_metadata_int(*model, "spkcache_sil_frames_per_spk");

    std::unordered_map<std::string, assets::onnx::Route> routes;
    for (const auto & [name, initializer] : model->initializers) {
        if (name.rfind("onnx::", 0) != 0) routes.emplace(name, assets::onnx::Route{name, false});
    }
    for (const auto & node : model->nodes) {
        if (node.op_type != "MatMul" || node.inputs.size() != 2 ||
            model->initializers.count(node.inputs[1]) == 0) {
            continue;
        }
        for (const auto & [pattern, replacement] : onnx_matmul_rules()) {
            if (std::regex_match(node.name, pattern)) {
                routes.emplace(std::regex_replace(node.name, pattern, replacement),
                               assets::onnx::Route{node.inputs[1], true});
                break;
            }
        }
    }
    for (const auto & [name, initializer] : model->initializers) {
        const bool routed = std::any_of(routes.begin(), routes.end(), [&](const auto & entry) {
            return entry.second.initializer == name;
        });
        if (!routed) {
            throw std::runtime_error("Nemotron 3 Diarization ONNX has an initializer no rule names: " + name);
        }
    }

    const auto sil_text = model->metadata.find("learnable_sil_emb");
    if (sil_text == model->metadata.end()) {
        throw std::runtime_error("Nemotron 3 Diarization ONNX has no learnable_sil_emb metadata: " + path.string());
    }
    std::vector<float> silence;
    std::stringstream sil_stream(sil_text->second);
    for (std::string item; std::getline(sil_stream, item, ',');) silence.push_back(std::stof(item));
    if (static_cast<int64_t>(silence.size()) != config.encoder.hidden_size) {
        throw std::runtime_error("Nemotron 3 Diarization ONNX learnable_sil_emb has the wrong size: " + path.string());
    }

    audio::MelFilterbankConfig mel;
    mel.sample_rate = features.sample_rate;
    mel.n_fft = features.n_fft;
    mel.n_mels = features.num_mel_bins;
    auto filterbank = audio::MelFilterbank().build(mel);
    for (float & value : filterbank.values) value = to_bf16(value);

    std::unordered_map<std::string, assets::onnx::Computed> computed;
    computed.emplace("preprocessor.fb", assets::onnx::Computed{filterbank.shape, std::move(filterbank.values)});
    computed.emplace("sortformer_modules.learnable_sil_emb",
                     assets::onnx::Computed{{config.encoder.hidden_size}, std::move(silence)});
    result->model_weights = assets::onnx::open_routed(model, std::move(routes), std::move(computed));
    result->mel_filterbank = result->model_weights->require_f32(
        "preprocessor.fb", {features.num_mel_bins, features.n_fft / 2 + 1});
    result->silence_embedding = result->model_weights->require_f32(
        "sortformer_modules.learnable_sil_emb", {config.encoder.hidden_size});
    result->frontend = std::make_shared<audio::NemoMelFrontend>(make_frontend(*result));
    return result;
}

}  // namespace

bool is_onnx_file(const std::filesystem::path & model_path) {
    return assets::onnx::is_file(model_path);
}

std::shared_ptr<const Assets> load_assets(
    const std::filesystem::path & model_root,
    const std::filesystem::path & spec_path) {
    if (is_onnx_file(model_root)) return load_onnx_assets(model_root);
    auto result = std::make_shared<Assets>();
    result->resources = engine::model_spec::load_resource_bundle(model_root, spec_path);
    result->model_config = parse_model_config(result->resources);
    result->feature_config = parse_feature_config(result->resources);
    result->model_weights = result->resources.open_tensor_source("weights");
    result->mel_filterbank = result->model_weights->require_f32(
        "preprocessor.fb",
        {result->feature_config.num_mel_bins, result->feature_config.n_fft / 2 + 1});
    result->silence_embedding = result->model_weights->require_f32(
        "sortformer_modules.learnable_sil_emb", {result->model_config.encoder.hidden_size});
    result->frontend = std::make_shared<audio::NemoMelFrontend>(make_frontend(*result));
    return result;
}

std::shared_ptr<ModelWeights> load_weights(
    const Assets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType storage_type,
    size_t context_bytes) {
    const auto & source = *assets.model_weights;
    const auto & config = assets.model_config;
    auto result = std::make_shared<ModelWeights>();
    result->store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "Nemotron 3 Diarization weights", context_bytes);
    auto & store = *result->store;
    const auto hidden = config.encoder.hidden_size;
    const auto intermediate = config.encoder.intermediate_size;
    const auto stacked = config.encoder.feature_size * config.encoder.subsampling_factor;
    result->pre_encode = modules::binding::linear_from_source(
        store, source, "encoder.pre_encode.proj", storage_type, hidden, stacked, false);
    result->embed_norm = modules::binding::norm_from_source(store, source, "encoder.embed_norm", hidden);
    result->layers.reserve(static_cast<size_t>(config.encoder.layers));
    for (int64_t index = 0; index < config.encoder.layers; ++index) {
        const std::string prefix = "encoder.layers." + std::to_string(index);
        EncoderLayerWeights layer;
        layer.norm1 = modules::binding::norm_from_source(store, source, prefix + ".norm1", hidden);
        layer.attention.qkv_weight = store.load_tensor(
            source, prefix + ".attn.w_qkv.weight", storage_type, {3 * hidden, hidden});
        layer.attention.out_weight = store.load_tensor(
            source, prefix + ".attn.out_proj.weight", storage_type, {hidden, hidden});
        layer.attention.out_bias = store.load_f32_tensor(
            source, prefix + ".attn.out_proj.bias", {hidden});
        layer.norm2 = modules::binding::norm_from_source(store, source, prefix + ".norm2", hidden);
        layer.ffn_in = modules::binding::linear_from_source(
            store, source, prefix + ".ffn.net.0", storage_type, intermediate, hidden, true);
        layer.ffn_out = modules::binding::linear_from_source(
            store, source, prefix + ".ffn.net.3", storage_type, hidden, intermediate, true);
        result->layers.push_back(std::move(layer));
    }
    result->final_norm = modules::binding::norm_from_source(store, source, "encoder.final_norm", hidden);
    result->encoder_projection = modules::binding::linear_from_source(
        store, source, "sortformer_modules.encoder_proj", storage_type,
        config.head.hidden_size, hidden, true);
    result->subpixel_upsample.weight = store.load_tensor(
        source, "sortformer_modules.subpixel_upsample.weight", storage_type,
        {config.head.hidden_size * config.head.upsample_factor,
         config.head.hidden_size, config.head.upsample_kernel_size});
    result->subpixel_upsample.bias = store.load_f32_tensor(
        source, "sortformer_modules.subpixel_upsample.bias",
        {config.head.hidden_size * config.head.upsample_factor});
    result->head_hidden = modules::binding::linear_from_source(
        store, source, "sortformer_modules.first_hidden_to_hidden", storage_type,
        config.head.hidden_size, config.head.hidden_size, true);
    result->speaker_head = modules::binding::linear_from_source(
        store, source, "sortformer_modules.single_hidden_to_spks", storage_type,
        config.num_speakers, config.head.hidden_size, true);
    store.upload();
    return result;
}

}  // namespace engine::models::nemotron_3_diar
