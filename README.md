# RLHF (Reinforcement Learning from Human Feedback) Library & Web Application

A comprehensive implementation of RLHF with a modern web interface for training and evaluating language models using human feedback.

## 🚀 Features

- **RLHF Library**: Complete Python implementation with PPO, DPO, and other RLHF algorithms
- **Web Interface**: Modern React.js frontend with Tailwind CSS
- **FastAPI Backend**: RESTful API for model training and inference
- **Docker Support**: Complete containerization with docker-compose
- **Real-time Training**: Live monitoring of training progress
- **Model Management**: Upload, train, and evaluate custom models
- **Human Feedback Collection**: Interactive interface for collecting human preferences

## 📁 Project Structure

```
rlhf-api/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Core configurations
│   │   ├── models/         # Data models
│   │   └── services/       # Business logic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               # React.js frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── hooks/          # Custom hooks
│   │   └── utils/          # Utility functions
│   ├── package.json
│   └── Dockerfile
├── rlhf_lib/              # RLHF Python library
│   ├── rlhf/
│   │   ├── algorithms/     # RLHF algorithms
│   │   ├── models/         # Model implementations
│   │   ├── data/           # Data processing
│   │   └── utils/          # Utilities
│   ├── setup.py
│   └── requirements.txt
├── docker-compose.yml      # Docker orchestration
├── docker-compose.dev.yml  # Development Docker setup
├── .env.example           # Environment variables
├── build_docker.sh        # Docker build script
├── dev_setup.sh           # Development setup script
├── quick_start.sh         # Quick start script
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

- Docker and Docker Compose (for production)
- Python 3.8+ (for development)
- Node.js 16+ (for development)

### Quick Start with Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rlhf-api
   ```

2. **Build and run with Docker**
   ```bash
   # Make the build script executable
   chmod +x build_docker.sh
   
   # Build and start all services
   ./build_docker.sh
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Celery Flower: http://localhost:5555

### Development Setup (Without Docker)

1. **Run the development setup**
   ```bash
   chmod +x dev_setup.sh quick_start.sh
   ./dev_setup.sh
   ```

2. **Start the services**
   ```bash
   # Quick start (both backend and frontend)
   ./quick_start.sh
   
   # Or start manually:
   # Terminal 1: Backend
   cd backend && python3 -m uvicorn app.main:app --reload
   
   # Terminal 2: Frontend
   cd frontend && npm start
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Manual Installation

#### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
cd frontend
npm install
npm start
```

#### RLHF Library Setup

```bash
cd rlhf_lib
pip install -e .
```

## 📚 Usage

### RLHF Library

```python
from rlhf import RLHFTrainer, PPOConfig, DPOConfig
from rlhf.models import GPT2Policy, GPT2Value
from rlhf.data import PreferenceDataset

# Initialize models
policy_model = GPT2Policy.from_pretrained("gpt2")
value_model = GPT2Value.from_pretrained("gpt2")

# Configure training
config = PPOConfig(
    learning_rate=1e-5,
    batch_size=4,
    max_grad_norm=1.0,
    target_kl=0.1
)

# Create trainer
trainer = RLHFTrainer(
    policy_model=policy_model,
    value_model=value_model,
    config=config
)

# Load preference data
dataset = PreferenceDataset.from_json("preferences.json")

# Train the model
trainer.train(dataset, epochs=10)
```

### Web Interface

1. **Upload Model**: Use the web interface to upload your base model
2. **Configure Training**: Set training parameters through the UI
3. **Collect Feedback**: Use the feedback interface to collect human preferences
4. **Monitor Training**: Watch real-time training progress
5. **Evaluate Results**: Compare model outputs and metrics

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/rlhf_db

# Redis
REDIS_URL=redis://localhost:6379

# Model Storage
MODEL_STORAGE_PATH=/app/models

# API Keys (optional)
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_TOKEN=your_hf_token
```

### Training Configuration

```python
# PPO Configuration
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=4,
    max_grad_norm=1.0,
    target_kl=0.1,
    gamma=1.0,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01
)

# DPO Configuration
dpo_config = DPOConfig(
    learning_rate=1e-5,
    batch_size=4,
    beta=0.1,
    max_grad_norm=1.0
)
```

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest

# Run frontend tests
cd frontend
npm test

# Run library tests
cd rlhf_lib
pytest
```

## 📊 Monitoring

The application includes comprehensive monitoring:

- **Training Metrics**: Loss curves, reward tracking, KL divergence
- **System Metrics**: GPU usage, memory consumption
- **API Metrics**: Request latency, error rates
- **Model Performance**: BLEU scores, human evaluation scores

## 🐳 Docker Commands

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Development setup (backend only)
docker-compose -f docker-compose.dev.yml up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Inspired by OpenAI's RLHF implementation
- Built with modern web technologies
- Uses state-of-the-art RLHF algorithms

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Join our community discussions 