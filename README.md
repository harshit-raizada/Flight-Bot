# Travel Assistant Chatbot

## Project Overview

This project implements a conversational AI chatbot designed to assist with travel-related inquiries. The chatbot can process detailed travel data, including flight schedules, layovers, and passenger details, and use this information to answer user questions. The system combines several key components to deliver an interactive, data-driven conversational experience, suitable for use in personal travel assistants or customer support in the travel industry.

## Features

- Natural Language Processing (NLP) with LangChain: The bot leverages LangChain's capabilities to handle text processing, retrieval, and memory, providing contextually relevant responses.
- FAISS Vector Store for Efficient Data Retrieval: FAISS (Facebook AI Similarity Search) enables fast and accurate retrieval of travel data, stored as text embeddings.
- Conversational Memory: Using LangChain's ConversationBufferMemory, the bot can maintain context within a conversation, allowing for a more coherent and natural interaction flow.
- Data-driven Approach: Processes structured travel data (in JSON format), storing it as text chunks that can be queried using an advanced embedding model.

## Project Structure

- initialize_chatbot: The main function for initializing the chatbot. It loads and processes travel data from a JSON file, splits the data into smaller text chunks, and creates a FAISS vector store for efficient data retrieval.
- process_travel_data: This function takes raw JSON travel data, validates its structure, and formats it into readable text chunks that can be used for vector storage.
- create_vector_store: Utilizes OpenAI's embeddings to convert processed text chunks into vectors and stores them in FAISS, allowing for rapid data retrieval.
- setup_chat_chain: Configures the chat chain with LangChain’s ConversationalRetrievalChain by combining the vector store and the conversation buffer memory, along with a custom prompt template.
- chat_with_bot: A function for querying the chatbot with a user question and retrieving a response.

## Output

![Demo](https://github.com/user-attachments/assets/e36a30c0-e7d1-482c-a190-3ae08e1327f3)
