from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import uuid
from rag_system import get_rag_system, get_conversation_manager

app = Flask(__name__)
CORS(app)

print("Starting Flask application...")


@app.route('/')
def index():
    """Return frontend page"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get or create session ID
        session_id = data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get conversation manager
        conv_manager = get_conversation_manager(session_id)
        
        # Generate response
        response = conv_manager.chat(message)
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset history."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        # Get conversation manager and reset
        try:
            conv_manager = get_conversation_manager(session_id)
            conv_manager.reset()
            return jsonify({'message': 'Conversation history reset'})
        except:
            return jsonify({'message': 'Session does not exist or has been reset'})
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'RAG chatbot service is running normally'})


if __name__ == '__main__':
    # Preload RAG system
    print("Preloading RAG system (this may take some time)...")
    get_rag_system()
    print("RAG system preload complete")
    
    # Start Flask application
    print("\nStarting Flask server...")
    print("Visit http://localhost:5000 to use the chat interface")
    app.run(debug=True, host='0.0.0.0', port=5000)

