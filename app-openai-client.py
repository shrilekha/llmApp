import os
from dotenv import load_dotenv

from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate

from flask import Flask, request, jsonify, render_template
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DT_ACCESS_TOKEN = os.getenv("DT_ACCESS_TOKEN")

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY") 
)

#Initialize Traceloop
try:
    headers = {"Authorization": f"Api-Token {DT_ACCESS_TOKEN}"}
    Traceloop.init(
        app_name="openai-llm-chat",
        api_endpoint="https://yex81559.sprint.dynatracelabs.com/api/v2/otlp",
        headers=headers
    )
    #print(f"Headers: {headers}")
    logger.info("Traceloop initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Traceloop: {e}")
    #print(f"DT_ACCESS_TOKEN: {DT_ACCESS_TOKEN}")


# Initialize Flask app
app = Flask(__name__)

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
        response = process_prompt_workflow(prompt_text, trace_id)
        #logger.debug(f"Full Result: {response}")
        
        #Extracting the required text from the response 
        content_text = response.output[0].content[0].text
        
        # Return response and trace ID
        return jsonify({"response": content_text, "traceparent": trace_id})  

    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@task(name="process_prompt")
def process_prompt_workflow(prompt_text, trace_id):
    logger.info(f"Starting workflow with trace ID: {trace_id}")
    response = generate_langchain_response(prompt_text, trace_id)
    return response

@task(name="generate_langchain_response")
def generate_langchain_response(prompt_text, trace_id):
    logger.info(f"Generating LangChain response for prompt: {prompt_text}")

    try:
        # Use the new OpenAI client to call the Chat Completions API
        response = client.responses.create(
            #model="gpt-3.5-turbo",  # Replace with the actual model you intend to use
            model="gpt-4o",
            input=prompt_text
        )
        logger.info("OpenAI response received successfully")
        print ("\nResponse:\n {response}")
        # One way is to extract the output text from the response, and send that alone. 
        # But that suppresses all the traceloop output params which are part of the payload
        # So return the entire response and process it later
        return response
    except Exception as e:
        logger.error(f"Error communicating with OpenAI API: {e}")
        raise


@task(name="generate_trace_id")
def generate_mock_trace_id():
    import uuid
    trace_id = f"00-{uuid.uuid4().hex[:32]}-0000000000000000-01"
    logger.debug(f"Generated mock trace ID: {trace_id}")
    return trace_id
        
if __name__ == "__main__":
    app.run(debug=False)
