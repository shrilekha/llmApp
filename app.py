import os
import openai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import requests

from flask import Flask, request, jsonify, render_template
import logging

# Environment variables
# Setup openai key
os.environ["OPENAI_API_KEY"] = "OPENAI-API-KEY"  // to update

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Initialize Traceloop
try:
    headers = {"Authorization":"Api-Token DT-TOKEN"}  // to update
    Traceloop.init(
        app_name="openai-llm-chat",
        api_endpoint="https://yex81559.sprint.dynatracelabs.com/api/v2/otlp",
        headers=headers
    )
    logger.info("Traceloop initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Traceloop: {e}")


# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set log level to DEBUG for detailed trace logs
logger = logging.getLogger(__name__)

TRACE_HEADER = "traceparent"  # Trace context header key

@app.route('/')
def index():
    logger.info("Rendering frontend UI")
    return render_template('index.html')

def forward_trace_id(headers, trace_id):
    headers[TRACE_HEADER] = trace_id
    logger.debug(f"Forwarding trace ID: {trace_id}")
    return headers

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        prompt_text = data.get('prompt')
        trace_id = request.headers.get(TRACE_HEADER)

        # Log received trace ID or fallback logic
        if trace_id:
            logger.info(f"Trace ID received = {trace_id}")
        else:
            logger.warning("No trace ID provided. Generating new trace ID.")
            trace_id = generate_mock_trace_id()

        # Call the main workflow with trace ID
        result = process_prompt_workflow(prompt_text, trace_id)
        logger.debug(f"Full Result: {result}")
        
        # Extract the content field
        content = result["choices"][0]["message"]["content"] if "choices" in result else "No response content"
        return jsonify({"response": str(content), "traceparent": trace_id})  # Return response and trace ID

    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@task(name="generate_trace_id")
def generate_mock_trace_id():
    import uuid
    trace_id = f"00-{uuid.uuid4().hex[:32]}-0000000000000000-01"  # Mock trace ID
    logger.debug(f"Generated mock trace ID: {trace_id}")
    return trace_id

@task(name="process_prompt")
def process_prompt_workflow(prompt_text, trace_id):
    logger.info(f"Starting workflow with trace ID: {trace_id}")
    response = generate_langchain_response(prompt_text, trace_id)
    return response

@task(name="generate_langchain_response")
def generate_langchain_response(prompt_text, trace_id):
    logger.info(f"Generating LangChain response for prompt: {prompt_text}")

    # Example downstream call to LangChain service
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer OPENAI-API-KEY" //update
    }
    headers = forward_trace_id(headers, trace_id)  # Attach trace ID
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt_text}]
    }

    try:
        logger.debug(f"Sending request to LangChain with trace ID: {trace_id}")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logger.info("LangChain response received successfully")
        return response.json()
    except Exception as e:
        logger.error(f"Error communicating with LangChain API: {e}")
        raise

if __name__ == "__main__":
    app.run(debug=False)
