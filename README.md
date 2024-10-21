

![Screenshot 2024-10-21 171639](https://github.com/user-attachments/assets/8903a815-9872-465b-a24a-dcab6c9dcd42)
![Screenshot 2024-10-21 171701](https://github.com/user-attachments/assets/a69a010f-c38f-4a83-a96b-f735dc37bb02)
![Screenshot 2024-10-21 171713](https://github.com/user-attachments/assets/3750ffa8-8770-47a1-93f1-37d381e09df7)

# SafeHaven-Mental_health_chatbot
this is a mental health chatbot which is built using llm and RAG pipeline
SafeHaven
SafeHaven is a mental health chatbot designed to provide support and information on various mental health issues such as anxiety and depression and many other mental issues. The chatbot leverages a state-of-the-art Large Language Model (LLM) and a Retrieval-Augmented Generation (RAG) pipeline to deliver accurate and helpful responses to user queries.

Features
Interactive Chatbot: Engage with the chatbot to ask questions related to mental health.
Information Retrieval: Utilizes external resources and documents to provide up-to-date information.
Personalized Suggestions: Offers tailored suggestions to help users cope with anxiety, depression, and other mental health challenges.
Document Similarity Search: Displays the most relevant documents based on user queries and their similarity scores.
User-Friendly Interface: Built with Streamlit for an intuitive user experience.







Technologies Used
1.Language Model: LLM based on the GROQ API for generating responses.
2.Retrieval-Augmented Generation (RAG): Combines retrieval of information and generative capabilities for comprehensive answers.
3.Document Loaders: Fetches information from PDF documents and web sources using PyPDFLoader.
4.Vector Store: Implements FAISS for efficient document retrieval.
5.Semantic Similarity: Utilizes SentenceTransformer for computing document similarity.
6.Web Scraping: Uses BeautifulSoup for extracting content from external websites.
7.Environment Management: Uses dotenv for managing environment variables.





Getting Started
To run the SafeHaven chatbot locally, follow these steps:

Prerequisites
Python 3.7 or later




Virtual environment (recommended)
Installation





1.Clone the repository:
git clone https://github.com/yourusername/SafeHaven.git
cd SafeHaven





2.Create a virtual environment (optional but recommended):





python -m venv venv





#mac/linux







source venv/bin/activate  

# On Windows use
venv\Scripts\activate





3.Install the required packages:
pip install -r requirements.txt





4. streamlit run app.py









 Set Up Environment Variables:






Create a .env file in the project root and add your Groq API key:
GROQ_API_KEY=your_groq_api_key_here





Usage




Simply enter your query about mental health, and SafeHaven will respond with informative answers and suggestions. For example, you can ask:

"How can I manage my anxiety?"
"What are the signs of depression?"
"Can you suggest techniques to overcome stress?"
The chatbot also provides a document similarity search feature, showing the most relevant documents related to your query along with their similarity scores.
Contact Information : atharvanarkhede0105@gmail.com


