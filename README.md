# 🎙️ Real-Time Voice AI Assistant

A sophisticated real-time voice conversation system that combines **speech-to-text**, **AI agents**, and **text-to-speech** for seamless voice interactions. Built with FastAPI, OpenAI Whisper, AutoGen, and Edge TTS.

## ✨ Features

### 🎯 **Core Capabilities**
- **Real-time Speech Recognition** - Live audio transcription using OpenAI Whisper
- **Intelligent AI Agents** - Multi-agent system with specialized roles (math, stories, general)
- **Natural Voice Synthesis** - High-quality text-to-speech with Microsoft Edge TTS
- **Continuous Conversation** - Automatic listening restart after responses
- **Web-based Interface** - Clean, responsive browser interface

### 🤖 **AI Agent System**
- **Manager Agent** - Routes queries to appropriate specialists
- **Math Agent** - Handles mathematical questions with kid-friendly explanations
- **Story Agent** - Creates engaging short stories for children
- **General Agent** - Manages greetings, casual conversation, and general queries
- **Fallback System** - Simple backup agent when AutoGen is unavailable

### 🔊 **Audio Processing**
- **Low-latency Streaming** - Real-time audio processing with WebSocket
- **Multiple Audio Formats** - WebM, Opus, WAV support
- **Voice Activity Detection** - Intelligent speech detection
- **Audio Quality Control** - Configurable chunk sizes and quality settings

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **FFmpeg** (for audio processing)
- **Microphone access** in your browser

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd whisper_streaming_web
```

2. **Create virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional)
Create a `.env` file:
```env
# For AutoGen AI agents (optional)
GEMINI_API_KEY=your_gemini_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gemini-1.5-flash
```

5. **Run the server**
```bash
python whisper_fastapi_online_server.py
```

6. **Open your browser**
Navigate to `http://localhost:8000`

## 🎮 Usage

### Basic Operation
1. **Click the microphone button** 🎙️ to start conversation
2. **Speak your question** clearly
3. **Listen to the AI response** with automatic voice synthesis
4. **Continue the conversation** - the system automatically restarts listening

### Example Interactions
- **Math**: "What is 2 plus 2?"
- **Stories**: "Tell me a story about a lion"
- **General**: "Hello, how are you?"

### Configuration Options
- **Chunk Size**: Adjust audio processing intervals (500ms - 5000ms)
- **WebSocket URL**: Change server endpoint if needed
- **Voice Selection**: Multiple high-quality neural voices available

## 🏗️ Architecture

### System Components
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │◄──►│   FastAPI Server │◄──►│   AI Agents     │
│                 │    │                  │    │                 │
│ • Microphone    │    │ • WebSocket      │    │ • AutoGen       │
│ • Audio Capture │    │ • Audio Pipeline │    │ • Gemini/OpenAI │
│ • TTS Playback  │    │ • Whisper STT    │    │ • Fallback      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Audio Pipeline
```
Microphone → WebM/Opus → FFmpeg → PCM → Whisper → Text → AI Agent → TTS → Audio
```

### File Structure
```
whisper_streaming_web/
├── whisper_fastapi_online_server.py  # Main server
├── requirements.txt                  # Dependencies
├── src/
│   ├── AI_agent/
│   │   └── ai_agent_runtime.py      # AutoGen agent system
│   ├── web/
│   │   └── live_transcription.html  # Frontend interface
│   ├── whisper_streaming/           # Whisper integration
│   └── diarization/                 # Speaker identification
└── README.md
```

## ⚙️ Configuration

### Command Line Options
```bash
python whisper_fastapi_online_server.py --help
```

Key parameters:
- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 8000)
- `--model`: Whisper model size (tiny.en, base, small, medium, large)
- `--min-chunk-size`: Audio chunk duration in seconds
- `--language`: Source language (en, es, fr, etc.)

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key for AI agents
- `OPENAI_API_KEY`: OpenAI API key (alternative to Gemini)
- `LLM_MODEL`: Model name (default: gemini-1.5-flash)

## 🔧 Advanced Features

### Multi-Agent System
The AI system uses AutoGen's multi-agent framework:
- **Intelligent Routing**: Manager agent analyzes queries and routes to specialists
- **Specialized Responses**: Each agent optimized for specific tasks
- **JSON Structured Output**: Consistent response formatting
- **Graceful Fallbacks**: Simple backup responses when needed

### Audio Quality Options
- **Multiple TTS Voices**: High-quality neural voices
- **Adaptive Chunk Sizes**: Balance between latency and accuracy
- **Voice Activity Detection**: Smart speech detection
- **Audio Format Support**: WebM, Opus, WAV compatibility

### Real-time Processing
- **WebSocket Streaming**: Low-latency bidirectional communication
- **Concurrent Processing**: Parallel audio and AI processing
- **State Management**: Conversation flow control
- **Error Recovery**: Robust error handling and recovery

## 🛠️ Development

### Key Dependencies
- **FastAPI**: Web framework and WebSocket support
- **OpenAI Whisper**: Speech recognition
- **AutoGen**: Multi-agent AI framework
- **Edge TTS**: Text-to-speech synthesis
- **FFmpeg**: Audio processing
- **PyTorch**: Machine learning backend

### Adding New Agents
1. Define agent in `src/AI_agent/ai_agent_runtime.py`
2. Add to manager's handoff list
3. Implement specialized system message
4. Test with conversation flow

### Customizing TTS
- Modify voice selection in `_edge_voices` list
- Adjust audio quality parameters
- Add new TTS backends as needed

## 🐛 Troubleshooting

### Common Issues
1. **Microphone not working**: Check browser permissions
2. **FFmpeg errors**: Ensure FFmpeg is installed and in PATH
3. **AI agent timeouts**: Check API keys and network connectivity
4. **Audio quality issues**: Adjust chunk size and audio format

### Debug Mode
Enable detailed logging by checking browser console and server output.

## 📄 License

This project is open source. Please check the license file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

**Built with ❤️ for seamless voice AI interactions**
