export const siteConfig = {
  name: "Aman Sharma",
  title: "Machine Learning Engineer",
  description: "Portfolio website of Aman Sharma",
  accentColor: "#1d4ed8",
  social: {
    email: "aman.sharma0206@gmail.com",
    linkedin: "https://www.linkedin.com/in/aman26sharma/",
    github: "https://github.com/Aman26Sharma/",
  },
  aboutMe:
    "I’m a Machine Learning Engineer with 2+ years of experience specializing in transformers, LLMs, and generative AI. At Huawei Canada, I work on optimizing large-scale language models for efficiency and real-world applications. I hold a Master’s in Data Science and AI from the University of Waterloo and a BEng. in Computer Engineering from Thapar University. I’m passionate about building practical AI solutions that turn cutting-edge research into impact.",
  skills: [
  "Python",
  "PyTorch",
  "TensorFlow",
  "Hugging Face Transformers",
  "LoRA / PEFT",
  "SQL",
  "Data Structures & Algorithms",
  "scikit-learn",
  "NumPy",
  "Pandas",
  "Matplotlib / Seaborn",
  "Git"
],
education: [
    {
      school: "University of Waterloo",
      degree: "Master of Data Science and Artificial Intelligence",
      dateRange: "Sept 2023 - Apr 2025",
      achievements: [
        "Graduated with GPA: 3.8/4.0",
        "Relevant Coursework: Machine Learning, Deep Learning, Computer Vision, Big Data Analytics, Stats for Data Science, Exploratory Data Analysis",
      ],
    },
    {
      school: "Thapar University",
      degree: "Bachelor of Engineering in Computer Engineering",
      dateRange: "2018 - 2022",
      achievements: [
        "Graduated with GPA 9.43/10 (Top 5% of class)",
      ],
    },
  ],
experience: [
    {
      company: "Huawei Canada",
      title: "Research Engineer - NLP/LLM",
      dateRange: "May 2024 - Present",
      bullets: [
        "Conducted layer-wise analysis of transformer architectures (embeddings, attention patterns) across scales, generating insights for architectural optimization",
        "Designed and evaluated architectural modifications to large-scale models, including LLaMA, Qwen, and Mixture of Experts (MoE), achieving 40% faster inference and 34% lower memory usage",
        "Integrated fused CUDA/Triton kernels into LLM training pipelines, delivering ∼30% faster training throughput and optimizing GPU utilization for large-scale experiments",
        "Leveraged PyTorch and HuggingFace Transformers for pre-training, fine-tuning (LoRA, PEFT, instruction tuning) and inference of large-scale language models (1B–40B parameters) in multi-node environments",
      ],
    },
    {
      company: "Samsung R&D India",
      title: "Software Engineer",
      dateRange: "Jul 2022 - Aug 2023",
      bullets: [
        "Collaborated to develop Push to Talk FW removing the need to hold the voice button while using voice feature in TV",
        "Integrated features in Text to Speech for non-zero ducking, making TV smarter to not reduce the background volume to zero while using Text to Speech",
        "Led a team to build the whole FW to integrate Speech to Text feature in Netflix App, enhancing user experience",
        "Contributed in Voice FW team to include the support of concurrent Multi-Voice Assistants and optimized performance of TTS and STT by 20%, by analyzing and resolving 50+ issues",
      ],
    },
    {
      company: "Samsung R&D India",
      title: "Machine Learning Intern",
      dateRange: "Jan 2022 - Jun 2022",
      bullets: [
        "Spearheaded end-to-end development of Textless NLP integration for Bixby voice assistant on Samsung TVs, achieving 25% reduction in voice command processing latency",
        "Architected and implemented a novel audio-to-pseudo-unit encoder, eliminating traditional speech-to-text conversion bottleneck and streamlining voice command processing pipeline",
        "Designed and trained a custom BERT-based classification model for direct mapping of voice commands to system functions, optimizing voice assistant response accuracy and efficiency",
      ],
    },
  ],
  publications: [
    {
      name: "DTRNet: Dynamic Token Routing Network to Reduce Quadratic Costs in Transformers",
      description:
        "arXiv preprint (2025); submitted to AAAI 2025 (under review)",
      link: "https://arxiv.org/abs/2509.00925",
    },
    {
      name: "Echoatt: Attend, copy, then adjust for more efficient large language models",
      description:
        " Efficient Natural Language and Speech Processing (ENLSP-IV) workshop, 2024",
      link: "https://arxiv.org/pdf/2409.14595",
    },
  ],
  projects: [
    {
      name: "DTRNet: Dynamic Token Routing Network to Reduce Quadratic Costs in Transformers",
      description:
        "DTRNet is an improved Transformer architecture that allows tokens to dynamically skip the quadratic cost of cross-token mixing while still receiving lightweight linear updates.",
      link: "https://github.com/Aman26Sharma/DTRNet",
      skills: ["PyTorch", "Python", "HuggingFace", "Distributed Training", "Model Optimization"],
    },
    {
      name: "Subquadratic Hybrid Transformer",
      description:
        "This project explores a hybrid transformer architecture that combines linear attention and sparse attention (BigBird) mechanisms to efficiently model long-range dependencies in long-sequence data.",
      link: "https://github.com/Aman26Sharma/Hybrid_LA_BB_transformer",
      skills: ["PyTorch", "Python", "HuggingFace", "LLMs"],
    },
    {
      name: "Intent Classification using GAN-BERT",
      description:
        "This project explores the application of semi-supervised learning for intent classification in the Chinese language using a GAN-BERT architecture.",
      link: "https://github.com/Aman26Sharma/GAN-BERT",
      skills: ["PyTorch", "Python", "HuggingFace", "GANs", "Semi-Supervised Learning"],
    },
    {
      name: "CNN on novel Human Actions Dataset",
      description:
        "In this project, we have tried to build our custom CNN model, which is then trained on a Human actions dataset also created by us",
      link: "https://github.com/Aman26Sharma/Custom-CNN-Model",
      skills: ["Python", "PyTorch", "CNN", "Computer Vision", "Dataset Creation & Annotation"],
    },
    {
      name: "Anime Recommendation System",
      description:
        "It is a content based Anime Recommendation. In this website when a user enters an Anime, it recommends top 10 animes of same genre based on user ratings",
      link: "https://github.com/Aman26Sharma/Anime-Recommendation",
      skills: ["Python", "Recommendation systems", "HTML/CSS", "Flask", "Pandas", "NumPy"],
    },
    {
      name: "Online Healthcenter",
      description:
        "In this project, I have developed a disease predicting app consisting of 5 diseases, trained using both Machine Learning(4 diseases) and Deep Learning(1 disease).",
      link: "https://github.com/Aman26Sharma/ML-Healthcenter",
      skills: ["Python", "ML", "HTML/CSS", "Flask", "Heroku"],
    },
    {
      name: "RNN Regularization",
      description:
        "This project reimplements and evaluates the core findings of Zaremba et al. (2014) by exploring dropout applied only to non-recurrent connections in RNNs.",
      link: "https://github.com/mbadalbadalian/RNN_Regularization",
      skills: ["RNN", "GRU", "Python"],
    },
  ],
  
};
