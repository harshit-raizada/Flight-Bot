# Importing libraries
import os
import sys
import json
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter 

# Load environment variables
load_dotenv()

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def format_datetime(datetime_str: str) -> str:
    """Convert ISO datetime string to more readable format"""
    dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
    return dt.strftime("%Y-%m-%d %H:%M %Z")

def process_travel_data(json_data: Dict) -> List[str]:
    """
    Convert travel JSON data into text chunks for vector storage
    """
    try:
        documents = []
        
        # Validate JSON structure
        if not json_data.get("user") or not json_data["user"].get("flights"):
            raise ValueError("No flight data found in JSON")
        
        for flight in json_data["user"]["flights"]:
            flight_info = (
                f"Booking reference (PNR): {flight['pnr']}\n"
                f"Travel class: {flight['class']}\n"
                f"Journey: {flight['source']} to {flight['destination']}\n"
                f"Departure: {format_datetime(flight['departure_date'])}\n"
                f"Arrival: {format_datetime(flight['arrival_date'])}\n"
                f"Layover duration: {flight['layover_duration']}\n"
            )
            documents.append(flight_info)
            
            # Process each flight segment
            for segment in flight["segments"]:
                segment_info = (
                    f"Flight {segment['flight_number']}\n"
                    f"From: {segment['departure']['airport']} ({segment['departure']['iata']})\n"
                    f"To: {segment['arrival']['airport']} ({segment['arrival']['iata']})\n"
                    f"Departure: {format_datetime(segment['departure']['date'])}\n"
                    f"Arrival: {format_datetime(segment['arrival']['date'])}\n"
                )
                documents.append(segment_info)
                
                # Process passenger information for each segment
                for passenger in segment["passengers"]:
                    passenger_info = (
                        f"Passenger {passenger['first_name']} {passenger['last_name']}\n"
                        f"Seat: {passenger['seat_number']}\n"
                        f"Baggage allowance:\n"
                        f"- Cabin: {passenger['cabin_baggage']}\n"
                        f"- Check-in: {passenger['check_in_baggage']}"
                    )
                    documents.append(passenger_info)

        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )
        
        return text_splitter.split_text("\n".join(documents))
    
    except KeyError as e:
        print(f"Error: Missing required field in JSON data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing travel data: {e}")
        sys.exit(1)

def create_vector_store(text_chunks: List[str]) -> FAISS:
    """
    Create a FAISS vector store from text chunks
    """
    try:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        return FAISS.from_texts(text_chunks, embeddings)
    except Exception as e:
        print(f"Error creating vector store: {e}")
        sys.exit(1)

def setup_chat_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """
    Set up the conversational chain with RAG
    """
    try:
        # Define the prompt template
        prompt_template = """
        You are a helpful travel assistant. Use the following context to answer questions 
        about the travel itinerary. Don't use any external knowledge. If you don't know something, just say so.

        Context: {context}

        Question: {question}

        Chat History: {chat_history}

        Answer: Let me help you with that.
        """

        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=prompt_template
        )

        # Setup memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create the chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-4", temperature=0.1),
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        return chain
    
    except Exception as e:
        print(f"Error setting up chat chain: {e}")
        sys.exit(1)

def initialize_chatbot(json_file_path: str) -> ConversationalRetrievalChain:
    """
    Initialize the chatbot with data from a JSON file
    """
    try:
        # Check if file exists
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Could not find file: {json_file_path}")
        
        # Load and process the data
        with open(json_file_path, 'r', encoding='utf-8') as file:
            travel_data = json.load(file)
        
        # Create text chunks
        text_chunks = process_travel_data(travel_data)
        
        # Create vector store
        vector_store = create_vector_store(text_chunks)
        
        # Setup and return the chat chain
        return setup_chat_chain(vector_store)
    
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {json_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)

def chat_with_bot(chat_chain: ConversationalRetrievalChain, question: str) -> str:
    """
    Get a response from the chatbot
    """
    try:
        return chat_chain.invoke({"question": question})['answer']
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def main():
    
    try:
        print("Initializing Travel Assistant...")
        chat_chain = initialize_chatbot("journey_details.json")
        print("Initialization complete!")
        
        # Interactive mode
        print("\nAsk your questions (type 'exit' to quit):")
        while True:
            try:
                user_question = input("\nYour question: ").strip()
                if user_question.lower() == 'exit':
                    print("Thank you for using the Travel Assistant!")
                    break
                if not user_question:
                    print("Please enter a question.")
                    continue
                    
                response = chat_with_bot(chat_chain, user_question)
                print(f"Answer: {response}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again or type 'exit' to quit.")
                
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()