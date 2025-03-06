### From https://medium.com/@phurlocker/building-an-ai-powered-registration-system-with-langgraph-fastapi-and-react-3e651ce83d71

from langgraph.graph import END
from db.sqlite_db import RegistrationState
from graph.base_graph import BaseGraphManager
import logging


class RegistrationGraphManager(BaseGraphManager):
    """Specialized graph manager with domain-specific (registration) logic."""

    def __init__(self, name: str, question_map: dict):
        # We pass RegistrationState as the state_class
        super().__init__(name, question_map, RegistrationState)

    def _build_graph(self):
        """
        Override the base build method to incorporate optional steps,
        or domain-specific edges.
        """

        def ask_question(state: RegistrationState, question_text: str):
            logging.info(f"[Registration] Transitioning to: {question_text}")
            return {
                "collected_data": state.collected_data,
                "current_question": question_text,
            }

        def create_question_node(question_text):
            return lambda s: ask_question(s, question_text)

        # Add each node
        for key, question_text in self.question_map.items():
            self.graph.add_node(key, create_question_node(question_text))

        # Set entry point
        self.graph.set_entry_point("ask_email")

        # Normal edges for the first two nodes
        self.graph.add_edge("ask_email", "ask_name")

        def path_func(state: RegistrationState):
            """Determine where to go next, considering multiple skips."""
            skip_address = state.collected_data.get("skip_ask_address", False)
            skip_phone = state.collected_data.get("skip_ask_phone", False)

            if skip_address and skip_phone:
                return "ask_username"  # Skip both address and phone
            elif skip_address:
                return "ask_phone"  # Skip address only
            else:
                return "ask_address"  # Default to asking for address first

        self.graph.add_conditional_edges(
            source="ask_name",
            path=path_func,
            path_map={
                "ask_address": "ask_address",
                "ask_phone": "ask_phone",
                "ask_username": "ask_username",
            },
        )

        self.graph.add_edge("ask_address", "ask_phone")
        self.graph.add_conditional_edges(
            source="ask_phone",
            path=lambda state: (
                "ask_username"
                if state.collected_data.get("skip_ask_phone", False)
                else "ask_username"
            ),
            path_map={"ask_username": "ask_username"},
        )
        self.graph.add_edge("ask_username", "ask_password")
        self.graph.add_edge("ask_password", END)

--------------------------------------------------------------------------
import guardrails as gd
import dspy
from validation.base_validator import BaseValidator
from pydantic import ValidationError
from validation.validated_response import ValidatedLLMResponse
from typing import Literal
import logging
import json
import mlflow
from helpers.config import OPENAI_API_KEY, MLFLOW_ENABLED, MLFLOW_EXPERIMENT_NAME

dspy.settings.configure(lm=dspy.LM(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY))

if MLFLOW_ENABLED:
    mlflow.dspy.autolog()

guard = gd.Guard.for_pydantic(ValidatedLLMResponse)

# Define DSPy Signature
class ValidateUserAnswer(dspy.Signature):
    """Validates and formats user responses. Should return 'valid', 'clarify', or 'error'."""

    question: str = dspy.InputField()
    user_answer: str = dspy.InputField()

    status: Literal["valid", "clarify", "error"] = dspy.OutputField(
        desc="Validation status: 'valid', 'clarify', or 'error'. Return 'valid' if the input contains all necessary information."
    )
    feedback: str = dspy.OutputField(
        desc="Explanation if response is incorrect or needs details."
    )
    formatted_answer: str = dspy.OutputField(
        desc="Return the response with proper formatting. Example: "
        "- Emails: Lowercase (e.g., 'John@gmail.com' → 'john@gmail.com'). "
        "- Names: Capitalize first & last name (e.g., 'john doe' → 'John Doe'). "
        "- Addresses: Capitalize & ensure complete info (e.g., '123 main st,newyork,ny' → '123 Main St, New York, NY 10001'). "
        "- Phone numbers: Format as (XXX) XXX-XXXX (e.g., '1234567890' → '(123) 456-7890'). "
        "An address must include: street number, street name, city, state, and ZIP code. "
        "Reject responses that do not meet this format with status='clarify'."
        "If the response cannot be formatted, return the original answer."
    )


run_llm_validation = dspy.Predict(ValidateUserAnswer)


class DSPyValidator(BaseValidator):
    """Uses DSPy with Guardrails AI for structured validation."""

    def validate(self, question: str, user_answer: str):
        """Validates user response, applies guardrails, and logs to MLflow."""
        try:
            raw_result = run_llm_validation(question=question, user_answer=user_answer)
            structured_validation_output = guard.parse(json.dumps(raw_result.toDict()))
            validated_dict = dict(structured_validation_output.validated_output)

            if MLFLOW_ENABLED:
                mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
                with mlflow.start_run(nested=True):
                    mlflow.log_param("validation_engine", "DSPy + Guardrails AI")
                    mlflow.log_param("question", question)
                    mlflow.log_param("input_answer", user_answer)
                    mlflow.log_param("status", validated_dict["status"])
                    mlflow.log_param(
                        "formatted_answer", validated_dict["formatted_answer"]
                    )

            return {
                "status": validated_dict["status"],
                "feedback": validated_dict["feedback"],
                "formatted_answer": validated_dict["formatted_answer"],
            }

        except ValidationError as ve:
            logging.error(f"Validation error: {str(ve)}")
            return {
                "status": "error",
                "feedback": "Output validation failed.",
                "formatted_answer": user_answer,
            }
        except Exception as e:
            logging.error(f"Validation error: {str(e)}")
            return {
                "status": "error",
                "feedback": "An error occurred during validation.",
                "formatted_answer": user_answer,
            }

-----------------------------------------------------------------------------
import openai
import guardrails as gd
import json
from typing import Dict
from validation.base_validator import BaseValidator
from validation.validated_response import ValidatedLLMResponse
from helpers.config import OPENAI_API_KEY, MLFLOW_ENABLED, MLFLOW_EXPERIMENT_NAME
import mlflow

if MLFLOW_ENABLED:
    mlflow.openai.autolog()

guard = gd.Guard.for_pydantic(ValidatedLLMResponse)


class ChatGPTValidator(BaseValidator):
    """ChatGPT-based implementation of the validation strategy."""

    def validate(self, question: str, user_answer: str) -> Dict[str, str]:
        """Uses OpenAI ChatGPT to validate responses."""
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that validates user responses. "
                        "You must respond in JSON format with a clear validation status. "
                        "If the response is valid, return: {'status': 'valid', 'feedback': '<feedback message>', 'formatted_answer': '<formatted response>'}. "
                        "If the response needs clarification, return: {'status': 'clarify', 'feedback': '<clarification message>', 'formatted_answer': '<original response>'}."
                        "Ensure proper formatting: lowercase emails, capitalized names, standardized phone numbers and addresses."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nUser Answer: {user_answer}\nValidate the answer.",
                },
            ],
            response_format={"type": "json_object"},
        )

        try:
            validation_str = response.choices[0].message.content.strip()
            validation_result = json.loads(validation_str)
            print(validation_result)

            # Apply Guardrails AI
            validated_result = guard.parse(json.dumps(validation_result))
            validated_dict = validated_result.validated_output
            print(validated_dict)
        except (json.JSONDecodeError, KeyError):
            validation_result = {
                "status": "error",
                "feedback": "Error processing validation response.",
                "formatted_answer": user_answer,
            }

        if MLFLOW_ENABLED:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(nested=True):
                mlflow.log_param("validation_engine", "ChatGPT")
                mlflow.log_param("question", question)
                mlflow.log_param("input_answer", user_answer)
                mlflow.log_param("status", validated_dict.get("status", "error"))
                mlflow.log_param(
                    "feedback", validated_dict.get("feedback", "No feedback")
                )
                mlflow.log_param(
                    "formatted_answer",
                    validated_dict.get("formatted_answer", user_answer),
                )

        return validated_dict

-----------------------------------------------------------------------------------------
from pydantic import BaseModel, Field, field_validator
import re

class ValidatedLLMResponse(BaseModel):
    """Validates & formats user responses using Guardrails AI & Pydantic."""

    status: str = Field(..., pattern="^(valid|clarify|error)$")
    feedback: str
    formatted_answer: str

    @field_validator("formatted_answer", mode="before")
    @classmethod
    def validate_and_format(cls, value, values):
        """Formats & validates responses based on the question type."""

        if values.get("status") == "error":
            return value  # Skip validation for errors

        question = values.get("question", "").lower()

        # Validate Email Format
        if "email" in question:
            return cls.validate_email(value)

        # Validate Name Format (Capitalize First & Last Name)
        if "name" in question:
            return cls.validate_name(value)

        # Validate Phone Number (Format: (XXX) XXX-XXXX)
        if "phone" in question:
            return cls.validate_phone(value)

        # Validate Address (Must contain street, city, state, ZIP)
        if "address" in question:
            return cls.validate_address(value)

        return value  # Default: Return unchanged

    @staticmethod
    def validate_email(email: str) -> str:
        """Validates email format and converts to lowercase."""
        return (
            email.lower() if re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email) else "clarify"
        )

    @staticmethod
    def validate_name(name: str) -> str:
        """Capitalizes first & last name."""
        return " ".join(word.capitalize() for word in name.split())

    @staticmethod
    def validate_phone(phone: str) -> str:
        """Validates & formats phone numbers as (XXX) XXX-XXXX."""
        digits = re.sub(r"\D", "", phone)  # Remove non-numeric characters
        return (
            f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            if len(digits) == 10
            else "clarify"
        )

    @staticmethod
    def validate_address(address: str) -> str:
        """Ensures address includes street, city, state, ZIP & formats correctly."""
        components = address.split(",")
        if len(components) < 3:
            return "clarify"  # Address is incomplete

        formatted_address = ", ".join(comp.strip().title() for comp in components)
        return (
            formatted_address if re.search(r"\d{5}", formatted_address) else "clarify"
        )

------------------------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uuid
import logging
from validation.factory import validate_user_input
from db.sqlite_db import fetch_session_from_db, upsert_session_to_db, RegistrationState
from graph.registration_graph import RegistrationGraphManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registration_questions = {
    "ask_email": "What is your email address?",
    "ask_name": "What is your full name?",
    "ask_address": "What is your address?",
    "ask_phone": "What is your phone number?",
    "ask_username": "Choose a username.",
    "ask_password": "Choose a strong password.",
}

registration_graph = RegistrationGraphManager("registration", registration_questions)
registration_graph.generate_mermaid_diagram()


@app.post("/start_registration")
def start_registration():
    session_id = str(uuid.uuid4())

    # Our initial state
    initial_state: RegistrationState = {
        "collected_data": {},
        "current_question": "",
        "current_node": "ask_email",
        "session_id": session_id,
    }

    # Start the graph & get the first node
    execution = registration_graph.compiled_graph.stream(initial_state)
    try:
        steps = list(execution)  # Fully consume the generator
        if not steps:
            raise RuntimeError("Generator exited before producing any states.")
        first_step = steps[0]
    except GeneratorExit:
        raise RuntimeError("GeneratorExit detected before first transition!")

    # Extract state from the first node
    first_node_key = list(first_step.keys())[0]
    first_node_state = first_step[first_node_key]
    first_node_state["current_node"] = first_node_key
    first_node_state["session_id"] = session_id

    # Save to session
    upsert_session_to_db(
        session_id,
        first_node_state["collected_data"],
        first_node_state["current_question"],
        first_node_state["current_node"],
    )

    return {
        "session_id": session_id,
        "message": first_node_state["current_question"],
        "state": first_node_state,
    }


@app.post("/submit_response")
def submit_response(response: dict):
    session_id = response.get("session_id")
    if not session_id:
        return {"error": "Missing session_id"}

    current_state = fetch_session_from_db(session_id)
    if not current_state:
        return {"error": "Session not found. Please restart registration."}

    skip_steps = response.get("skip_steps", [])
    for node_key in skip_steps:
        logging.info(f"skip_{node_key}")
        current_state[f"skip_{node_key}"] = True

    user_answer = response.get("answer", "")
    current_question = current_state["current_question"]
    current_node = current_state["current_node"]

    # Use dspy to validate the answer with fallbacks
    if current_node in skip_steps:
        # If user is skipping this question, create a dummy validation result
        validation_result = {
            "status": "valid",
            "feedback": "Skipped this question",
            "formatted_answer": "-",
        }
        logging.info(f"Skipping validation for {current_node}")
    else:
        # Normal validation
        validation_result = validate_user_input(current_question, user_answer)

        # If there's a clarify/error
        if validation_result["status"] in ("clarify", "error"):
            return {
                "next_question": current_question,
                "validation_feedback": validation_result["feedback"],
                "user_answer": user_answer,
                "formatted_answer": validation_result["formatted_answer"],
                "state": current_state,
            }

    current_state["collected_data"][current_state["current_node"]] = validation_result[
        "formatted_answer"
    ]

    if "current_node" not in current_state or not current_state.get("collected_data"):
        return {"error": "Corrupt session state, restart registration."}

    next_step = registration_graph.resume_and_step_graph(current_state)

    if not next_step or next_step == {}:
        # Means we've hit the END node or no more steps
        return {
            "message": "Registration complete!",
            "validation_feedback": validation_result["feedback"],
            "user_answer": user_answer,
            "formatted_answer": validation_result["formatted_answer"],
            "state": current_state,
            "summary": current_state["collected_data"],
        }

    next_node_key = list(next_step.keys())[0]
    next_node_state = next_step[next_node_key]
    next_node_state["current_node"] = next_node_key

    upsert_session_to_db(
        session_id,
        current_state["collected_data"],
        next_node_state["current_question"],
        next_node_state["current_node"],
    )

    return {
        "next_question": next_node_state["current_question"],
        "validation_feedback": validation_result["feedback"],
        "user_answer": user_answer,
        "formatted_answer": validation_result["formatted_answer"],
        "state": next_node_state,
        "summary": current_state["collected_data"],
    }


@app.post("/edit_field")
def edit_field(request: dict):
    session_id = request.get("session_id")
    if not session_id:
        return {"error": "Missing session_id"}

    field_to_edit = request.get("field_to_edit")
    new_value = request.get("new_value")

    current_state = fetch_session_from_db(session_id)
    if not current_state:
        logging.error("Session not found. Please restart registration.")
        return {"error": "Session not found. Please restart registration."}

    question_text = registration_questions.get(field_to_edit)
    if not question_text:
        logging.error(f"Invalid field_to_edit: {field_to_edit}")
        return {"error": f"Invalid field_to_edit: {field_to_edit}"}

    validation_result = validate_user_input(
        question=question_text, user_answer=new_value
    )

    if validation_result["status"] == "clarify":
        return {
            "message": "Needs clarification",
            "validation_feedback": validation_result["feedback"],
            "raw_answer": new_value,
            "formatted_answer": validation_result["formatted_answer"],
        }

    current_state["collected_data"][field_to_edit] = validation_result[
        "formatted_answer"
    ]

    upsert_session_to_db(
        session_id,
        current_state["collected_data"],
        current_state["current_question"],
        current_state["current_node"],
    )

    return {
        "message": "Field updated successfully!",
        "validation_feedback": validation_result["feedback"],
        "raw_answer": new_value,
        "formatted_answer": validation_result["formatted_answer"],
        "summary": current_state["collected_data"],
    }

