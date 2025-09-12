export const siteConfig = {
  name: "Sushant Gautam",
  title: "AI/ML Engineer • Building AI Agents",
  description: "Portfolio of Sushant Gautam — AI/ML Engineer focused on agentic systems, MLOps, and scalable deployments.",
  accentColor: "#008073",
  social: {
    email: "mail.sushantgautam@icloud.com",
    linkedin: "https://www.linkedin.com/in/susan-gautam/",
    github: "https://github.com/sushant097/",
  },

  aboutMe:
    "AI/ML engineer and ex-SDE Intern at AWS. I build agentic AI systems end to end—data pipelines, model training and evaluation, and production deployment on AWS. I’ve shipped computer vision and LLM/RAG projects in healthcare and autonomy, and I care about scalability, observability, and cost. I like turning messy logs and research papers into simple, useful products. Currently exploring robust AI agents, evaluation harnesses, and scalable MLOps on AWS. I’m also seeking a full-time role where I can build and ship reliable AI.",

  skills: [
    "Python", "PyTorch", "TensorFlow", "Scikit-learn",
    "LLMs", "RAG", "Agentic AI", "LangChain", "LLMOps",
    "Computer Vision", "GANs", "Diffusion Models",
    "MLOps", "Docker", "Kubernetes", "MLflow", "DVC",
    "Airflow", "Kafka", "PySpark", "AWS", "GCP", "CI/CD",
    "Next.js", "TypeScript", "FastAPI", "SQL"
  ],

  projects: [
    {
      name: "🩺 Medical RAG Chatbot",
      description:
        "A Retrieval-Augmented Generation (RAG) medical chatbot that answers user queries using a curated medical knowledge base. Combines LLMs with vector search (LangChain + Pinecone) to deliver grounded, context-aware answers instead of relying purely on model memory. Demonstrates end-to-end AI: data ingestion, embeddings, retrieval, UI (Flask), and AWS CI/CD deployment.",
      link: "https://github.com/sushant097/medical-rag-chatbot/",
      skills: ["LangChain", "Pinecone", "CI/CD", "AWS", "LLMs", "RAG"],
    },
    {
      name: "End-to-End MLOps: Image Classification",
      description:
        "Production-style ML pipeline with training, evaluation, CI/CD, and deployment on AWS SageMaker.",
      link: "https://github.com/sushant097/ML-Interview-Preparation/tree/master/ML-Interview-Preparation/MlOps-Project/End-to-End-Mlops-Image-Classification-Project",
      skills: ["Python", "PyTorch", "SageMaker", "MLflow", "DVC", "AWS"]
    },
    {
      name: "Real-Time Data Streaming Pipelines",
      description:
        "Ingest, transform, and serve streams using Airflow, Kafka, PySpark, Docker, and Cassandra.",
      link: "https://github.com/sushant097/ML-Interview-Preparation/tree/master/ML-Interview-Preparation/MlOps-Project/End-to-End-Data-Streaming",
      skills: ["Airflow", "Kafka", "PySpark", "Cassandra", "Docker"]
    },
    {
      name: "Self-Driving Car (End-to-End DL)",
      description:
        "Minimal autonomous driving stack: steering angle prediction from front-camera frames.",
      link: "https://github.com/sushant097/Self-Driving-Car-Projects",
      skills: ["PyTorch", "Computer Vision", "OpenCV"]
    },
    {
      name: "Handwritten Line Text Recognition (OCR)",
      description:
        "CRNN (CNN+LSTM+CTC) OCR achieving low CER on IAM; data aug + real-time demo.",
      link: "https://github.com/sushant097/Handwritten-Line-Text-Recognition-using-Deep-Learning-with-Tensorflow",
      skills: ["TensorFlow", "OCR", "Computer Vision"]
    },
    {
      name: "ROS2 Autonomous Tractor",
      description:
        "Sensor fusion with LIDAR, GPS, and RGB-D; control stack for autonomous navigation.",
      link: "https://github.com/sushant097/Ros2-Autonomous-Tractor",
      skills: ["ROS2", "Perception", "Controls"]
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
    {
      school: "Tribhuvan University",
      degree: "B.E. in Computer Engineering, GPA 4.0",
      dateRange: "2015 – 2019",
      achievements: [
        "Graduated with distinction; merit scholarship",
        "Competitions & research projects in CV/ML"
      ]
    }
  ]
} as const;
