export const siteConfig = {
  name: "Sushant Gautam",
  title: "Software Engineer | MLOps | Gen AI",
  description: "Portfolio of Sushant Gautam — AI/ML Engineer focused on agentic systems, MLOps, and scalable deployments.",
  accentColor: "#008073",
  social: {
    email: "mail.sushantgautam@icloud.com",
    linkedin: "https://www.linkedin.com/in/susan-gautam/",
    github: "https://github.com/sushant097/",
  },

  aboutMe:
    "Software Engineer with a passion for building intelligent and generative AI systems that scale. Experienced in LLMs, RAG, LangChain, and MLOps (Docker, Kubernetes, CI/CD), with deep expertise in computer vision, deep learning, and cloud deployments. I enjoy architecting robust data pipelines, optimizing models, and delivering reliable AI products that solve real-world problems. Open to full-time roles in software engineering, AI/ML or MLOps.",

  skills: [
    "Python", "Java", "C++", "CUDA", "AutoML", "TypeScript", "Next.js", "FastAPI", "SQL",
    "PyTorch", "TensorFlow", "Scikit-learn", "Computer Vision", "GANs", "Diffusion Models",
    "LLMs", "GenAI", "RAG", "Agentic AI", "LangChain", "LLMOps", "Hugging Face",
    "MLOps", "Docker", "Kubernetes", "MLflow", "DVC", "Airflow", "Kafka", "PySpark",
    "AWS", "GCP", "CI/CD", "Jenkins", "Prometheus", "Grafana",
    "Cassandra", "ROS2", "Git", "Linux", "Data Engineering"
  ],

  projects: [
    {
      name: "🧠 Context-Aware RAG Memory",
      description:
        "Persistent semantic memory system that learns from web reading. Uses hybrid temporal-semantic retrieval to recall exact snippets, integrated with a Chrome extension for highlight-on-recall.",
      link: "https://github.com/sushant097/RAG-Memory-Context-Aware-AI-Agent",
      skills: ["FastAPI", "Gemini 2.0 Flash/Ollama", "FAISS Vector Store", "MCP Tools", "MarkItDown", "Chrome MV3", "Local/Cloud Embeddings"]
    },
    {
      name: "🧩 Hybrid Agent Planner",
      description:
        "High-reliability agent prioritizing correctness. Fuses LLM reasoning with heuristic guardrails and secure Python sandboxing. Features a Jaccard-based semantic cache to instantly recall historical answers, bypassing the planning loop for recurring queries.",
      link: "https://github.com/sushant097/Hybrid-Agent-Planner",
      skills: ["Gemini 2.0 / Ollama", "MultiMCP Dispatcher", "Jaccard Semantic Cache", "Dynamic Python Planning", "Heuristic Guardrails", "Runtime Sandbox"]
    },
    {
      name: "🤖 LLM Agents: External APIs & RAG",
      description:
        "Cortex-style agentic framework bridging LLMs to the real world. Orchestrates a Perceive → Plan → Act loop using MCP to manipulate Google Sheets, Gmail, and Web Search via a Telegram gateway.",
      link: "https://github.com/sushant097/LLM-Agents-External-APIs",
      skills: ["Gemini 2.0 Flash", "MCP (Stdio/SSE)", "Google OAuth 2.0", "Trafilatura", "MarkItDown", "FAISS", "Telegram Bot API"]
    },
    {
      name: "📈🤖 AI Stock Advisor",
      description:
        "Streamlit-powered app for real-time stock analysis and company research. Integrates Yahoo Finance data, LLM agents, and Gemini for personalized investment reports and actionable insights.",
      link: "https://github.com/sushant097/ai-stock-advisor",
      skills: ["Python", "Streamlit", "LLMs", "Gemini", "Agentic AI", "yfinance", "Data Engineering", "Prompt Engineering"]
    },
    {
      name: "🧩⚡ TabGist — Chrome Summarizer Extension",
      description:
        "Chrome MV3 extension with FastAPI backend for summarizing web pages and YouTube transcripts into concise, print-ready notes. Features local-first processing, Gemini-powered output, and multi-language support.",
      link: "https://github.com/sushant097/tabgist-chrome",
      skills: ["JavaScript", "Chrome Extension", "FastAPI", "Python", "LLMs", "Gemini", "NLP", "REST APIs"]
    },
    {
      name: "🩺 Medical RAG Chatbot",
      description:
        "End-to-end Retrieval-Augmented Generation chatbot for medical Q&A. Uses LLMs, LangChain, and Pinecone for vector search over curated medical data, with Flask UI and AWS CI/CD deployment. Delivers context-grounded, reliable answers.",
      link: "https://github.com/sushant097/medical-rag-chatbot/",
      skills: ["Python", "LLMs", "RAG", "LangChain", "Pinecone", "Flask", "Vector Search", "Embeddings", "AWS", "CI/CD"]
    },
    {
      name: "End-to-End MLOps: Image Classification",
      description:
        "Robust ML pipeline for image classification: data ingestion, training, evaluation, and automated CI/CD deployment on AWS SageMaker. Includes MLflow tracking and DVC for reproducibility.",
      link: "https://github.com/sushant097/ML-Interview-Preparation/tree/master/ML-Interview-Preparation/MlOps-Project/End-to-End-Mlops-Image-Classification-Project",
      skills: ["Python", "PyTorch", "SageMaker", "MLflow", "DVC", "AWS", "MLOps", "CI/CD", "Data Engineering"]
    },
    {
      name: "Real-Time Data Streaming Pipelines",
      description:
        "Scalable pipelines for real-time data ingestion, transformation, and serving. Built with Airflow orchestration, Kafka streaming, PySpark processing, Docker containers, and Cassandra storage.",
      link: "https://github.com/sushant097/ML-Interview-Preparation/tree/master/ML-Interview-Preparation/MlOps-Project/End-to-End-Data-Streaming",
      skills: ["Python", "Airflow", "Kafka", "PySpark", "Docker", "Cassandra", "Data Engineering", "Streaming"]
    },
    {
      name: "Self-Driving Car (End-to-End DL)",
      description:
        "Autonomous driving stack for steering angle prediction using deep learning on front-camera frames. Includes data preprocessing, model training, and real-time inference with OpenCV.",
      link: "https://github.com/sushant097/Self-Driving-Car-Projects",
      skills: ["Python", "PyTorch", "Computer Vision", "OpenCV", "Deep Learning", "Autonomous Vehicles"]
    },
    {
      name: "Handwritten Line Text Recognition (OCR)",
      description:
        "Handwritten text recognition using CRNN (CNN+LSTM+CTC) architecture. Achieved low CER on IAM dataset with advanced data augmentation and real-time demo interface.",
      link: "https://github.com/sushant097/Handwritten-Line-Text-Recognition-using-Deep-Learning-with-Tensorflow",
      skills: ["Python", "TensorFlow", "OCR", "Computer Vision", "Deep Learning", "Data Augmentation"]
    },
    {
      name: "ROS2 Autonomous Tractor",
      description:
        "Autonomous tractor navigation using sensor fusion (LIDAR, GPS, RGB-D) and ROS2. Developed control stack for real-time perception, localization, and path planning in outdoor environments.",
      link: "https://github.com/sushant097/Ros2-Autonomous-Tractor",
      skills: ["Python", "ROS2", "Sensor Fusion", "Perception", "Controls", "Localization", "Path Planning"]
    }
  ],

  experience: [
    {
      company: "Amazon (AWS) — Transcribe Streaming",
      title: "SDE Intern",
      dateRange: "May 2025 – Jul 2025",
      bullets: [
        "Built a Next.js/TypeScript dashboard summarizing 1,000s of integration test runs in <5s.",
        "Zero-touch deployments with AWS CDK + Elastic Beanstalk; prod rollout in <10 min.",
        "Failure summarization reduced root-cause analysis time by ~40–60%."
      ]
    },
    {
      company: "Mississippi State University",
      title: "Graduate Research Assistant (CV/GenAI)",
      dateRange: "Aug 2023 – May 2025",
      bullets: [
        "Image restoration research (GANs, diffusion) with Stanford/Princeton/BlueHalo; +20% quality.",
        "CI/CD for ML (Docker, K8s, Airflow); training/eval with PyTorch-Lightning, Optuna."
      ]
    },
    {
      company: "Proxmed Pty. Ltd.",
      title: "Machine Learning Engineer",
      dateRange: "Feb 2023 – Aug 2023",
      bullets: [
        "LVO detection with YOLOv7; +8% accuracy; serverless deploy via AWS Lambda.",
        "Model optimization (quantization/ONNX) and CI/CD (GitLab, Docker, K8s)."
      ]
    },
    {
      company: "Wiseyak / HeHealth (prior roles)",
      title: "ML/CV Engineer & AI Research Scientist (Contract)",
      dateRange: "Oct 2020 – Apr 2022",
      bullets: [
        "Medical imaging models; reduced false positives and improved diagnostic accuracy.",
        "Synthetic data (GANs) + few-shot learning boosted model accuracy."
      ]
    }
  ],

  education: [
    {
      school: "Mississippi State University",
      degree: "M.S. in Computer Science (AI), GPA 4.0",
      dateRange: "Aug 2023 – Aug 2025",
      achievements: [
        "Coursework: Algorithms, Cloud Computing, AI Robotics, Autonomous Vehicles, Data Science, Deep Learning",
        "Thesis: X-DECODE — EXtreme Deblurring with Curriculum Optimization and Domain Equalization (arXiv, 2025). https://arxiv.org/abs/2504.08072",
        "President, CSE AI Club; organized GenAI workshops and hackathons"
      ]
    },
    // {
    //   school: "Tribhuvan University",
    //   degree: "B.E. in Computer Engineering, GPA 4.0",
    //   dateRange: "2015 – 2019",
    //   achievements: [
    //     "Graduated with distinction; merit scholarship",
    //     "Competitions & research projects in CV/ML"
    //   ]
    // }
  ]
} as const;
