export const siteConfig = {
  name: "Sushant Gautam",
  title: "AI/ML Engineer | Agentic Systems & MLOps",
  description: "Portfolio of Sushant Gautam — AI/ML Engineer focused on building scalable Agentic AI systems, RAG architectures, and production MLOps pipelines.",
  accentColor: "#008073",
  social: {
    email: "mail.sushantgautam@icloud.com",
    linkedin: "https://www.linkedin.com/in/susan-gautam/",
    github: "https://github.com/sushant097/",
  },

  aboutMe:
    "I am an AI/ML Engineer who owns the entire model lifecycle—from translating business ambiguity into concrete mathematical problems to deploying scalable, low-latency systems in production. My approach combines rigorous research (MS in CS, 4.0 GPA) with production discipline (AWS, CI/CD). Whether architecting 'System 2' agentic workflows or optimizing inference costs for serverless deployments, I focus on building robust, observable AI products that solve real problems at scale.",
  
    // Optimized Skill Order: GenAI -> Core ML -> MLOps -> Cloud -> Languages -> Tools
  skills: [
    // GenAI & Agents (The Hook)
    "Agentic AI", "LLMs", "RAG", "LangChain", "Model Context Protocol (MCP)", "Generative AI",
    
    // Core ML & Deep Learning
    "PyTorch", "Computer Vision", "Deep Learning", "Diffusion Models", "GANs", "Scikit-learn", "TensorFlow",
    
    // MLOps & Engineering
    "MLOps", "LLMOps", "Docker", "Kubernetes", "MLflow", "DVC", "Airflow", "CI/CD",
    
    // Cloud & Infrastructure
    "AWS", "SageMaker", "AWS CDK", "GCP", "BASH", "Terraform",
    
    // Languages & Backend
    "Python", "FastAPI", "SQL", "TypeScript", "Next.js", "C++", "CUDA",
    
    // Data & Tools
    "Data Engineering", "Kafka", "PySpark", "Git", "Vector Databases", "Prometheus", "Grafana"
  ],

  projects: [
    {
      name: "🧩 Cognitive-Reflex-Agent",
      description:
        "A 'System 2' reasoning agent capable of writing and executing Python code in a secure sandbox to solve complex queries. Features a semantic caching layer ('System 1') to recall past executions, significantly reducing latency and inference costs.",
      link: "https://github.com/sushant097/Cognitive-Reflex-Agent",
      skills: [
        "Agentic AI",
        "Model Context Protocol (MCP)",
        "Code Interpreter",
        "Security Sandboxing",
        "Semantic Caching",
        "Gemini 2.0"
      ]
    },
    {
      name: "🧠 Context-Aware RAG Memory",
      description:
        "A persistent memory system that 'learns' from user browsing. Uses hybrid temporal-semantic retrieval to recall exact information snippets, integrated with a Chrome extension for real-time augmentation.",
      link: "https://github.com/sushant097/RAG-Memory-Context-Aware-AI-Agent",
      skills: ["RAG", "Vector Databases", "FastAPI", "Chrome Extensions", "Embeddings", "Ollama"]
    },
    {
      name: "📈🤖 AI Stock Advisor",
      description:
        "Multi-agent financial analyst platform. Orchestrates specialized LLM agents to fetch real-time Yahoo Finance data, analyze market trends, and generate personalized investment reports.",
      link: "https://github.com/sushant097/ai-stock-advisor",
      skills: ["Multi-Agent Systems", "Python", "Streamlit", "Financial Data Engineering", "Prompt Engineering"]
    },
    {
      name: "🧩⚡ TabGist — Chrome Summarizer",
      description:
        "Production-ready Chrome Extension for summarizing web content and YouTube transcripts. Built with a local-first philosophy using Gemini Flash for sub-second latency.",
      link: "https://github.com/sushant097/tabgist-chrome",
      skills: ["Chrome MV3", "JavaScript", "FastAPI", "NLP", "GenAI", "REST APIs"]
    },
    {
      name: "🩺 Medical RAG Chatbot",
      description:
        "HIPAA-conscious RAG system for medical Q&A. retrieval accuracy optimized using LangChain and Pinecone, deployed via AWS CI/CD pipelines for reliable 24/7 availability.",
      link: "https://github.com/sushant097/medical-rag-chatbot/",
      skills: ["RAG", "LangChain", "Pinecone", "AWS", "Flask", "Healthcare AI"]
    },
    {
      name: "End-to-End MLOps Pipeline",
      description:
        "Full-cycle MLOps implementation for image classification. Automates data versioning (DVC), experiment tracking (MLflow), and model deployment to AWS SageMaker endpoints.",
      link: "https://github.com/sushant097/ML-Interview-Preparation/tree/master/ML-Interview-Preparation/MlOps-Project/End-to-End-Mlops-Image-Classification-Project",
      skills: ["MLOps", "AWS SageMaker", "MLflow", "DVC", "GitHub Actions", "PyTorch"]
    },
    {
      name: "Real-Time Data Streaming Architecture",
      description:
        "High-throughput data engineering pipeline processing live streams. Orchestrates Kafka producers/consumers and PySpark transformations within Dockerized Airflow environments.",
      link: "https://github.com/sushant097/ML-Interview-Preparation/tree/master/ML-Interview-Preparation/MlOps-Project/End-to-End-Data-Streaming",
      skills: ["Data Engineering", "Apache Kafka", "Airflow", "PySpark", "Cassandra", "Docker"]
    },
    {
      name: "Self-Driving Car (End-to-End DL)",
      description:
        "Autonomous steering system using Convolutional Neural Networks (CNNs). Clones human driving behavior by predicting steering angles directly from raw camera pixels.",
      link: "https://github.com/sushant097/Self-Driving-Car-Projects",
      skills: ["Deep Learning", "Computer Vision", "OpenCV", "PyTorch", "Autonomous Vehicles"]
    },
    {
      name: "Handwritten Text Recognition (OCR)",
      description:
        "Custom CRNN (CNN + LSTM + CTC Loss) architecture for reading handwriting. Achieved state-of-the-art accuracy on the IAM dataset using advanced data augmentation techniques.",
      link: "https://github.com/sushant097/Handwritten-Line-Text-Recognition-using-Deep-Learning-with-Tensorflow",
      skills: ["TensorFlow", "OCR", "Sequence Modeling", "Computer Vision", "Data Augmentation"]
    },
    {
      name: "ROS2 Autonomous Tractor",
      description:
        "Navigation stack for agricultural robotics. Implements sensor fusion (LIDAR + GPS) and path planning algorithms using ROS2 for precise field traversal.",
      link: "https://github.com/sushant097/Ros2-Autonomous-Tractor",
      skills: ["ROS2", "Robotics", "Sensor Fusion", "Localization", "Path Planning"]
    }
  ],

 experience: [
    {
      company: "Amazon Web Services (AWS)",
      title: "SDE Intern (Transcribe Team)",
      dateRange: "May 2025 – Jul 2025",
      bullets: [
        "Architected a Next.js/TypeScript observability dashboard to visualize 1,000s of integration tests, reducing debug time by 50%.",
        "Engineered a zero-touch CI/CD pipeline using AWS CDK & Elastic Beanstalk, slashing deployment time from hours to <10 minutes.",
        "Developed AI-driven failure summarization tools, accelerating root-cause analysis by 40% for the engineering team."
      ]
    },
    {
      company: "Mississippi State University",
      title: "Graduate Research Assistant",
      dateRange: "Aug 2023 – May 2025",
      bullets: [
        "Collaborated with researchers from **Stanford & Princeton** to develop diffusion models for image restoration, improving quality by 20%.",
        "Built reproducible ML training pipelines using Docker, Kubernetes, and PyTorch-Lightning, optimizing hyperparameter tuning with Optuna."
      ]
    },
    {
      company: "Proxmed Pty. Ltd.",
      title: "Machine Learning Engineer",
      dateRange: "Feb 2023 – Aug 2023",
      bullets: [
        "Deployed LVO detection models (YOLOv7) on AWS Lambda (Serverless), achieving 8% higher accuracy than legacy systems.",
        "Optimized model inference using ONNX quantization and automated deployments via GitLab CI/CD."
      ]
    },
    {
      company: "Wiseyak / HeHealth",
      title: "ML Engineer & Researcher",
      dateRange: "Oct 2020 – Apr 2022",
      bullets: [
        "Developed medical imaging diagnosis models, leveraging synthetic data generation (GANs) to overcome data scarcity.",
        "Reduced false positive rates in skin lesion detection through few-shot learning techniques."
      ]
    }
  ],

  education: [
    {
      school: "Mississippi State University",
      degree: "M.S. in Computer Science (AI), GPA 4.0",
      dateRange: "Aug 2023 - Aug 2025",
      achievements: [
        "**Thesis:** X-DECODE — EXtreme Deblurring with Curriculum Optimization (arXiv, 2025).",
        "**Leadership:** President, CSE AI Club (Organized GenAI workshops & hackathons).",
        "**Coursework:** Cloud Computing, AI Robotics, Deep Learning, Advanced Algorithms."
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
