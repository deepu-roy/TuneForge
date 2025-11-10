# Changelog

## [0.1.0] - 2025-11-10

### Added

- 🎉 Initial release of TuneForge
- 🔨 Full-featured CLI with modular commands:
  - `tuneforge pipeline` - Complete workflow
  - `tuneforge train` - Train models with LoRA
  - `tuneforge merge` - Merge adapters with base models
  - `tuneforge convert` - Convert to GGUF format
  - `tuneforge test` - Test fine-tuned models
- 📝 Configuration file support (config.env)
- 🤖 Auto-generated Ollama Modelfile
- 🍎 Apple Silicon (MPS) support
- ⚡ Modern APIs (SFTConfig, no deprecation warnings)
- 📦 Flexible quantization support (f16, f32, q8_0, q4_0, etc.)
- 🔧 Shell script alternative (train-and-convert.sh)
- 📚 Comprehensive documentation

### Features

- LoRA-based efficient fine-tuning
- Automatic pipeline orchestration
- Skip steps for partial reruns
- Multiple quantization versions
- Ready-to-deploy Ollama integration

### Supported

- Python 3.12+
- PyTorch with MPS (Apple Silicon) or CPU
- TinyLlama and compatible models
- GGUF conversion via llama.cpp
