🎥 Fake Buster — SIH 2024

🧠 Deep-Fake Detection using Machine Learning & CNN

📖 Overview

Fake Buster is a machine-learning project developed for the Smart India Hackathon 2024 (SIH 2024).
It aims to detect deep-fake media (images/videos) using Convolutional Neural Networks (CNNs).

This system is designed to identify manipulated content efficiently and can be adapted for real-world applications in media authentication, cybersecurity, and digital forensics.

🚀 Key Features

✅ Deep-fake detection using CNNs
✅ Real-time web interface for testing media
✅ Dataset preprocessing & feature extraction pipeline
✅ Flask-based backend for deployment
✅ Responsive HTML/CSS UI
✅ Organized, modular codebase

🧩 Tech Stack
Category	Tools/Technologies
Programming Language	Python 🐍
Framework	Flask
Libraries	NumPy, OpenCV, TensorFlow/Keras, Pandas
Frontend	HTML, CSS, JavaScript
Environment	Jupyter Notebook, VS Code
📁 Project Structure
Fake_Buster_SIH_2024/
├── static/                # CSS, JS, images for web interface  
├── templates/             # HTML templates  
├── uploads/               # Uploaded media files for testing  
├── model-version-1.ipynb  # CNN model development notebook  
├── app.py                 # Flask application entry point  
├── requirement.txt        # Dependencies  
├── LICENSE                # Apache 2.0 License  
└── README.md              # Project documentation  

⚙️ Installation & Setup
# 1️⃣ Clone this repository
git clone https://github.com/Salmali-Chattopadhyay/Fake_Buster_SIH_2024.git

# 2️⃣ Navigate into the project folder
cd Fake_Buster_SIH_2024

# 3️⃣ Install required dependencies
pip install -r requirement.txt

# 4️⃣ Run the application
python app.py


Then open your browser and visit:
👉 http://127.0.0.1:5000

🧠 Model Details

Convolutional Neural Network (CNN) for image feature extraction

Preprocessing: frame extraction, normalization, resizing

Fully connected layers for binary classification (Real vs Fake)

Dataset: mix of authentic and manipulated images/videos

Trained with high validation accuracy (update with your results!)

🎯 Future Improvements

Add video-level detection instead of frame-level

Integrate fake-voice detection for multimedia validation

Deploy model using Docker or on cloud (AWS, Heroku, GCP)

Add explainability layer (Grad-CAM) for visualization

📜 License

This project is licensed under the Apache 2.0 License — see the LICENSE
 file for details.

🙌 Acknowledgements

Special thanks to Smart India Hackathon 2024 organizers, mentors, and team members for their support and guidance.

💡 Developed with passion by Salmali Chattopadhyay
🗓️ October 2025
🌐 For learning, innovation, and real-world problem solving.
