#include "engine/models/nemotron_asr/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"
#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/assets/transcribe_cpp_gguf.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_asr {
namespace json = engine::io::json;
namespace {

void validate_config(const NemotronConfig & config) {
    if (config.model_type != "nemotron3_5_asr") {
        throw std::runtime_error("Nemotron ASR expects model_type=nemotron3_5_asr");
    }
    if (config.vocab_size <= 0 || config.blank_token_id < 0 || config.blank_token_id >= config.vocab_size) {
        throw std::runtime_error("Nemotron ASR invalid vocab metadata");
    }
    if (config.decoder_hidden_size <= 0 || config.decoder_layers != 2) {
        throw std::runtime_error("Nemotron ASR currently expects the 2-layer RNNT decoder from the checkpoint");
    }
    if (config.encoder.hidden_size <= 0 || config.encoder.layers <= 0 || config.encoder.heads <= 0 ||
        config.encoder.hidden_size % config.encoder.heads != 0) {
        throw std::runtime_error("Nemotron ASR invalid encoder metadata");
    }
    if (config.encoder.subsampling_factor != 8 || config.encoder.subsampling_kernel != 3 ||
        config.encoder.subsampling_stride != 2) {
        throw std::runtime_error("Nemotron ASR expects the official factor-8 causal subsampling config");
    }
    if (config.frontend.sample_rate != 16000 || config.frontend.feature_size <= 0 ||
        config.frontend.n_fft <= 0 || config.frontend.win_length <= 0 || config.frontend.hop_length <= 0) {
        throw std::runtime_error("Nemotron ASR invalid frontend metadata");
    }
    if (config.prompt_dictionary.empty()) {
        throw std::runtime_error("Nemotron ASR processor_config prompt dictionary is empty");
    }
}

std::vector<uint8_t> parse_special_token_ids(const std::filesystem::path & tokenizer_json, int64_t vocab_size) {
    std::vector<uint8_t> special(static_cast<size_t>(vocab_size), 0);
    const auto root = json::parse_file(tokenizer_json);
    if (const auto * added = root.find("added_tokens"); added != nullptr && added->is_array()) {
        for (const auto & item : added->as_array()) {
            if (!json::optional_bool(item, "special", false)) {
                continue;
            }
            const int64_t id = json::require_i64(item, "id");
            if (id >= 0 && id < vocab_size) {
                special[static_cast<size_t>(id)] = 1;
            }
        }
    }
    return special;
}

void parse_decoder_metadata(
    const std::filesystem::path & tokenizer_json,
    std::string & metaspace_replacement,
    bool & trim_leading_space) {
    const auto root = json::parse_file(tokenizer_json);
    metaspace_replacement.clear();
    trim_leading_space = false;
    if (const auto * decoder = root.find("decoder"); decoder != nullptr
        && decoder->is_object()
        && decoder->find("type") != nullptr
        && decoder->require("type").as_string() == "Metaspace") {
        metaspace_replacement = decoder->require("replacement").as_string();
        trim_leading_space = true;
    }
}

NemotronConfig parse_config(const assets::ResourceBundle & resources) {
    const auto config_root = resources.parse_json("config");
    const auto processor_root = resources.parse_json("processor_config");

    NemotronConfig config;
    config.model_type = json::require_string(config_root, "model_type");
    config.vocab_size = json::require_i64(config_root, "vocab_size");
    config.blank_token_id = json::require_i64(config_root, "blank_token_id");
    config.pad_token_id = json::require_i64(config_root, "pad_token_id");
    config.default_prompt_id = json::require_i64(config_root, "default_prompt_id");
    config.decoder_hidden_size = json::require_i64(config_root, "decoder_hidden_size");
    config.decoder_layers = json::require_i64(config_root, "num_decoder_layers");
    config.max_symbols_per_step = json::require_i64(config_root, "max_symbols_per_step");
    config.num_prompts = json::require_i64(config_root, "num_prompts");
    config.prompt_intermediate_size = json::require_i64(config_root, "prompt_intermediate_size");

    const auto & encoder = config_root.require("encoder_config");
    config.encoder.hidden_size = json::require_i64(encoder, "hidden_size");
    config.encoder.intermediate_size = json::require_i64(encoder, "intermediate_size");
    config.encoder.layers = json::require_i64(encoder, "num_hidden_layers");
    config.encoder.heads = json::require_i64(encoder, "num_attention_heads");
    config.encoder.kv_heads = json::require_i64(encoder, "num_key_value_heads");
    config.encoder.conv_kernel = json::require_i64(encoder, "conv_kernel_size");
    config.encoder.sliding_window = json::require_i64(encoder, "sliding_window");
    config.encoder.subsampling_factor = json::require_i64(encoder, "subsampling_factor");
    config.encoder.subsampling_channels = json::require_i64(encoder, "subsampling_conv_channels");
    config.encoder.subsampling_kernel = json::require_i64(encoder, "subsampling_conv_kernel_size");
    config.encoder.subsampling_stride = json::require_i64(encoder, "subsampling_conv_stride");
    config.encoder.max_position_embeddings = json::require_i64(encoder, "max_position_embeddings");
    config.encoder.default_lookahead_tokens =
        json::require_i64(encoder, "default_num_lookahead_tokens");
    config.encoder.supported_lookahead_tokens =
        json::require_i64_array(encoder, "supported_num_lookahead_tokens");

    const auto & feature = processor_root.require("feature_extractor");
    config.frontend.sample_rate = json::require_i64(feature, "sampling_rate");
    config.frontend.feature_size = json::require_i64(feature, "feature_size");
    config.frontend.n_fft = json::require_i64(feature, "n_fft");
    config.frontend.win_length = json::require_i64(feature, "win_length");
    config.frontend.hop_length = json::require_i64(feature, "hop_length");
    config.frontend.preemphasis = json::require_f32(feature, "preemphasis");
    config.encoder.default_lookahead_tokens =
        json::require_i64(processor_root, "default_num_lookahead_tokens");
    config.encoder.supported_lookahead_tokens =
        json::require_i64_array(processor_root, "supported_num_lookahead_tokens");
    config.prompt_dictionary = json::require_i64_object(processor_root, "prompt_dictionary");
    validate_config(config);
    return config;
}

// transcribe.cpp names of the Nemotron subsampling (NeMo's Sequential: conv, act, depthwise, pointwise, act, ...)
// and of its language prompt head; its conformer, predictor and joint use the shared naming.
const assets::transcribe_cpp::NameRules & transcribe_cpp_rules() {
    static const auto rules = assets::transcribe_cpp::compile_rules({
        {R"(^enc\.pre_encode\.conv\.0\.(weight|bias)$)", "encoder.subsampling.conv_in.$1"},
        {R"(^enc\.pre_encode\.conv\.2\.(weight|bias)$)", "encoder.subsampling.layers.0.depthwise_conv.$1"},
        {R"(^enc\.pre_encode\.conv\.3\.(weight|bias)$)", "encoder.subsampling.layers.0.pointwise_conv.$1"},
        {R"(^enc\.pre_encode\.conv\.5\.(weight|bias)$)", "encoder.subsampling.layers.1.depthwise_conv.$1"},
        {R"(^enc\.pre_encode\.conv\.6\.(weight|bias)$)", "encoder.subsampling.layers.1.pointwise_conv.$1"},
        {R"(^enc\.pre_encode\.out\.(weight|bias)$)", "encoder.subsampling.linear.$1"},
        {R"(^prompt\.mlp\.0\.(weight|bias)$)", "prompt_projector.linear_1.$1"},
        {R"(^prompt\.mlp\.2\.(weight|bias)$)", "prompt_projector.linear_2.$1"},
    });
    return rules;
}

std::shared_ptr<const NemotronASRAssets> load_transcribe_cpp_assets(const std::filesystem::path & path) {
    const assets::GgufMetadata metadata(path);
    auto assets = std::make_shared<NemotronASRAssets>();
    auto & config = assets->config;
    config.model_type = "nemotron3_5_asr";
    config.vocab_size = metadata.require_int("stt.parakeet.predictor.vocab");
    config.blank_token_id = metadata.require_int("tokenizer.ggml.blank_token_id");
    config.pad_token_id = metadata.require_int("tokenizer.ggml.unknown_token_id");
    config.default_prompt_id = metadata.require_int("stt.parakeet.prompt.auto_id");
    config.decoder_hidden_size = metadata.require_int("stt.parakeet.predictor.hidden");
    config.decoder_layers = metadata.require_int("stt.parakeet.predictor.n_layers");
    config.num_prompts = metadata.require_int("stt.parakeet.prompt.num_prompts");
    config.prompt_intermediate_size = metadata.require_int("stt.parakeet.prompt.hidden");

    auto & enc = config.encoder;
    enc.hidden_size = metadata.require_int("stt.parakeet.encoder.d_model");
    enc.intermediate_size = metadata.require_int("stt.parakeet.encoder.d_ff");
    enc.layers = metadata.require_int("stt.parakeet.encoder.n_layers");
    enc.heads = metadata.require_int("stt.parakeet.encoder.n_heads");
    enc.kv_heads = enc.heads;
    enc.conv_kernel = metadata.require_int("stt.parakeet.encoder.conv_kernel");
    // the window counts the current frame besides its left context
    enc.sliding_window = metadata.require_int("stt.parakeet.encoder.att_context_left") + 1;
    enc.subsampling_factor = metadata.require_int("stt.parakeet.encoder.subsampling_factor");
    enc.subsampling_channels = metadata.require_int("stt.parakeet.encoder.subsampling_channels");
    enc.max_position_embeddings = metadata.require_int("stt.parakeet.encoder.pos_emb_max_len");
    enc.default_lookahead_tokens = metadata.require_int("stt.parakeet.encoder.att_context_right");
    const auto choices = metadata.require_int_array("stt.parakeet.encoder.att_context_size_choices");
    if (choices.empty() || choices.size() % 2 != 0) {
        throw std::runtime_error("transcribe.cpp Nemotron GGUF has malformed att_context_size_choices: " + path.string());
    }
    enc.supported_lookahead_tokens.clear();
    for (size_t i = 1; i < choices.size(); i += 2) enc.supported_lookahead_tokens.push_back(choices[i]);

    config.frontend.sample_rate = metadata.require_int("stt.frontend.sample_rate");
    config.frontend.feature_size = metadata.require_int("stt.frontend.num_mels");
    config.frontend.n_fft = metadata.require_int("stt.frontend.n_fft");
    config.frontend.win_length = metadata.require_int("stt.frontend.win_length");
    config.frontend.hop_length = metadata.require_int("stt.frontend.hop_length");
    config.frontend.preemphasis = metadata.require_float("stt.frontend.pre_emphasis");

    const auto locales = metadata.require_string_array("stt.parakeet.prompt.dictionary.locales");
    const auto indices = metadata.require_int_array("stt.parakeet.prompt.dictionary.indices");
    if (locales.size() != indices.size()) {
        throw std::runtime_error("transcribe.cpp Nemotron GGUF prompt dictionary sizes differ: " + path.string());
    }
    for (size_t i = 0; i < locales.size(); ++i) config.prompt_dictionary.emplace(locales[i], indices[i]);

    auto tokens = metadata.require_string_array("tokenizer.ggml.tokens");
    const auto token_types = metadata.require_int_array("tokenizer.ggml.token_type");
    if (static_cast<int64_t>(tokens.size()) != config.vocab_size || token_types.size() != tokens.size()) {
        throw std::runtime_error("transcribe.cpp Nemotron GGUF vocabulary does not match predictor.vocab: " + path.string());
    }
    // GGUF token types: 1 is a normal piece, the rest (unknown, control) are special
    assets->special_token_ids.assign(tokens.size(), 0);
    for (size_t i = 0; i < token_types.size(); ++i) assets->special_token_ids[i] = token_types[i] != 1 ? 1 : 0;
    assets->metaspace_replacement = "\xE2\x96\x81";
    assets->trim_leading_space = true;
    assets->tokenizer = std::make_shared<engine::tokenizers::HuggingFaceTokenizerJson>(
        std::move(tokens), assets->metaspace_replacement, assets->trim_leading_space);

    assets->source = assets::transcribe_cpp::open_renamed(path, transcribe_cpp_rules());
    const auto conv_in = assets->source->require_metadata("encoder.subsampling.conv_in.weight");
    if (conv_in.shape.size() != 4) {
        throw std::runtime_error("transcribe.cpp Nemotron GGUF has an unexpected subsampling conv: " + path.string());
    }
    enc.subsampling_kernel = conv_in.shape[3];
    enc.subsampling_stride = 2;
    validate_config(config);
    return assets;
}

}  // namespace

bool is_transcribe_cpp_file(const std::filesystem::path & model_path) {
    return assets::transcribe_cpp::is_file(model_path, "rnnt");
}

std::shared_ptr<const NemotronASRAssets> load_nemotron_asr_assets(const std::filesystem::path & model_path) {
    if (is_transcribe_cpp_file(model_path)) return load_transcribe_cpp_assets(model_path);
    auto resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("nemotron_asr"));
    auto assets = std::make_shared<NemotronASRAssets>();
    assets->resources = std::move(resources);
    assets->source = assets->resources.open_tensor_source("weights");
    assets->config = parse_config(assets->resources);
    const auto & tokenizer_json = assets->resources.require_file("tokenizer_json");
    assets->tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(tokenizer_json);
    assets->special_token_ids =
        parse_special_token_ids(tokenizer_json, assets->config.vocab_size);
    parse_decoder_metadata(
        tokenizer_json,
        assets->metaspace_replacement,
        assets->trim_leading_space);
    return assets;
}

}  // namespace engine::models::nemotron_asr
