import torch
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support
from app import chatbot_function  # Import chatbot function from app.py

# Load the embedding model used in the chatbot
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate embedding vector for given text."""
    return embedding_model.encode(text, convert_to_tensor=True)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two embedding vectors."""
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()

def evaluate_chatbot(repeat_tests=5, threshold=0.7):
    """Evaluate chatbot based on response consistency and coherence."""
    
    questions = [
        "What are the symptoms of diabetes?",
        "How can I treat a fever at home?",
        "What are the causes of high blood pressure?",
        "Explain the role of insulin in the body.",
        "What are the benefits of regular exercise?"
    ]

    total_similarity = 0
    y_true = []
    y_pred = []
    
    print("\nStarting chatbot evaluation...\n")
    
    for question in questions:
        print(f"Question: {question}")

        responses = []
        for _ in range(repeat_tests):
            chatbot_response = chatbot_function(question)
            responses.append(chatbot_response)
            time.sleep(1)  # Avoid overwhelming the chatbot
        
        # Convert responses to embeddings
        response_embeddings = [get_embedding(resp) for resp in responses]

        # Calculate pairwise similarity between responses
        similarities = []
        for i in range(len(response_embeddings)):
            for j in range(i + 1, len(response_embeddings)):
                similarity = cosine_similarity(response_embeddings[i], response_embeddings[j])
                similarities.append(similarity)

        # Compute average similarity score
        avg_similarity = sum(similarities) / len(similarities)
        total_similarity += avg_similarity
        
        # Classify as consistent/inconsistent
        y_true.append(1)  # 1 = expected consistency
        y_pred.append(1 if avg_similarity >= threshold else 0)  # 1 = consistent
        
        print(f"  → Avg Consistency Score: {avg_similarity:.2f}\n")
    
    # Compute overall evaluation metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    final_consistency_score = total_similarity / len(questions)

    print("\n=== Chatbot Evaluation Results ===")
    print(f"Final Consistency Score: {final_consistency_score:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return final_consistency_score, precision, recall, f1

if __name__ == "__main__":
    evaluate_chatbot()
