"""
Title: o1-reasoning Pipeline powered by OpenAI
Author: Victor Carvalho Tavernari
Date: 2024-09-18
Version: 0.1-alpha
License: MIT
Description:
    This pipeline processes user messages and interacts with the OpenAI API.
    It breaks down user input into steps, processes each step, and composes a final answer based on the steps and their results.
Requirements: openai, pydantic
"""

from pydantic import BaseModel, Field, ConfigDict
import os
import subprocess
import json
import logging
from typing import List, Union, Generator, Iterator
from openai import OpenAI
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
)

# Create a custom formatter


class ColorFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log levels.
    """
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m'  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)


formatter = ColorFormatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Setting up the logging configuration
logger = logging.getLogger("o1-reasoning")  # Using your logger's name
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# MARK: Models


class CodeExecution(BaseModel):
    """
    Model representing a code execution request.
    """
    code: str = Field(
        ...,
        description="""
        The code to be executed in Python.
        To answer the step, you must add print statements to show the result of the code execution.
        """,
        strict=True,
    )

    model_config = ConfigDict(
        extra='forbid',
        strict=True,
    )


class Step(BaseModel):
    """
    Model representing a single step in the chain of thoughts.
    """
    title: str = Field(
        ...,
        description="The title of the step.",
    )

    description: str = Field(
        ...,
        description="The description of the step.",
    )

    model_config = ConfigDict(
        extra='forbid',
    )


class ChainOfThoughts(BaseModel):
    """
    Model representing the chain of thoughts as a list of steps.
    """
    steps: List[Step] = Field(
        ...,
        description="The steps of the chain of thoughts.",
    )

    model_config = ConfigDict(
        extra='forbid',
    )


class AnswerEvaluation(BaseModel):
    """
    Model representing the evaluation of the answer.
    """
    is_correct: bool = Field(
        ...,
        description="Whether the answer is correct.",
    )

# MARK: Pipeline


class Pipeline:
    """
    Pipeline for processing user messages and interacting with the OpenAI API.

    This pipeline breaks down user input into steps, processes each step,
    and composes a final answer based on the steps and their results.
    """

    class Valves(BaseModel):
        """
        Configuration class for storing API keys and model settings.

        Attributes:
            OPENAI_API_KEY (str): The API key for OpenAI services.
            MODEL_NAME (str): The name of the OpenAI model to use.
            OPENAI_BASE_URL (Optional[str]): The base URL for OpenAI services.
            MAX_CHAIN_OF_THOUGHTS_EXECUTION (int): The maximum number of chain of thoughts executions allowed.
        """

        OPENAI_API_KEY: str = ""
        MODEL_NAME: str = "gpt-4o-mini"
        OPENAI_BASE_URL: str | None = None
        MAX_CHAIN_OF_THOUGHTS_EXECUTION: int = 2

    def __init__(self):
        """
        Initializes the Pipeline instance by setting up configurations and OpenAI client.
        """
        self.name = "o1-reasoning"
        self.valves = self.Valves(
            OPENAI_API_KEY=os.getenv(
                "OPENAI_API_KEY", "your-openai-api-key-here"),
            MODEL_NAME="gpt-4o-mini",
            OPENAI_BASE_URL=None,
            MAX_CHAIN_OF_THOUGHTS_EXECUTION=1,
        )
        self.client = None  # Will be initialized in on_startup
        self.is_title_generation = False
        logger.debug(
            "Pipeline initialized with configuration: %s", self.valves)

        self.total_of_chain_of_thoughts_executions = 0

    async def on_startup(self):
        """
        Called when the server is started. Initializes the OpenAI client.
        """
        logger.info("Pipeline startup initiated.")
        OPENAI_API_KEY = self.valves.OPENAI_API_KEY
        OPENAI_BASE_URL = self.valves.OPENAI_BASE_URL

        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        logger.debug("OpenAI client initialized.")

    async def on_shutdown(self):
        """
        Called when the server is stopped. Performs any necessary cleanup.
        """
        logger.info("Pipeline shutdown initiated.")

    async def inlet(self, body: dict, user: dict) -> dict:
        logger.info("inlet: %s", __name__)

        metadata = body.get("metadata", {})
        task = metadata.get("task", "")
        self.is_title_generation = task == "title_generation"
        self.total_of_chain_of_thoughts_executions = 0

        return body

    async def on_valves_updated(self):
        OPENAI_API_KEY = self.valves.OPENAI_API_KEY
        OPENAI_BASE_URL = self.valves.OPENAI_BASE_URL

        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

    def execute_python_code(self, code: str) -> (str, int):
        """
        Executes the given Python code and returns the output and return code.

        Args:
            code (str): The Python code to execute.

        Returns:
            Tuple[str, int]: A tuple containing the combined output and the return code.
        """
        logger.debug("(execute_python_code) Executing Python code:\n%s", code)
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                check=True,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            combined_output = stdout + ("\n" + stderr if stderr else "")
            logger.debug(
                "(execute_python_code) Code execution result: %s",
                combined_output
            )
            return combined_output, result.returncode
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip()
            logger.error(
                "(execute_python_code) Code execution failed: %s",
                stderr
            )
            return stderr, e.returncode
        except Exception as e:
            logger.exception("(execute_python_code) Code execution failed.")
            return str(e), 1

    def code_execution_tool(self) -> ChatCompletionToolParam:
        code_execution_schema = CodeExecution.model_json_schema()
        code_execution_schema["additionalProperties"] = False
        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name="code_execution",
                description="Execute the code to help to answer the step using Python. You must add print statements to show the result of the code execution.",
                parameters=code_execution_schema,
                strict=True,
            ),
            type="function",
        )

    def answer_step(self, step: Step, messages: List[dict]) -> str:
        """
        Processes a single step by interacting with the OpenAI API and handling any code execution.

        Args:
            step (Step): The step to process.
            messages (List[dict]): The message history for the conversation.

        Returns:
            str: The content of the final answer for the step.
        """
        logger.info("(answer_step) Answering step: %s", step.title)

        tools = [self.code_execution_tool()]

        messages = messages + [
            ChatCompletionUserMessageParam(
                content=(
                    f"Step: {step.title}"
                    f"Description: {step.description}"
                    "You must follow the step proposal to provide your answer."
                ),
                role="user",
            ),
        ]

        logger.debug("(answer_step) messages: %s", messages)

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.valves.MODEL_NAME,
            temperature=0.15,
            tools=tools,
        )

        message = response.choices[0].message
        logger.debug(
            "(answer_step) message: %s", message.content
        )
        messages.append(message)

        if response.choices[0].finish_reason == "tool_calls":
            for tool_call in response.choices[0].message.tool_calls:
                logger.debug("Tool call received: %s", tool_call)
                if tool_call.function.name == "code_execution":
                    try:
                        json_args = json.loads(tool_call.function.arguments)
                        code = json_args.get("code")
                        logger.debug("Code to execute:\n%s", code)
                        code_execution_result, code_execution_return = self.execute_python_code(
                            code)

                        if code_execution_return != 0:
                            error_message = f"Error: {code_execution_result}"
                            logger.error(
                                "Code execution failed: %s", error_message)
                            messages.append(
                                ChatCompletionToolMessageParam(
                                    tool_call_id=tool_call.id,
                                    content=error_message,
                                    role="tool",
                                )
                            )
                            continue

                        messages.append(
                            ChatCompletionToolMessageParam(
                                tool_call_id=tool_call.id,
                                content=code_execution_result,
                                role="tool",
                            )
                        )
                    except json.JSONDecodeError as e:
                        logger.error("JSON decode error: %s", e)
                        continue

            messages.append(
                ChatCompletionUserMessageParam(
                    content=(
                        "You must write the final answer based on the context and the steps."
                        "Write each step details and answers."
                        "Write a final report based on the context and the steps."
                    ),
                    role="user",
                ),
            )

            response = self.client.chat.completions.create(
                messages=messages,
                model=self.valves.MODEL_NAME,
                temperature=0.15,
            )
            messages.append(response.choices[0].message)
            logger.debug("(answer_step) tools response: %s", response)

        content = response.choices[0].message.content
        logger.info("(answer_step) Step '%s' answer: %s", step.title, content)
        return content

    def execute_chain_of_thoughts(self, chain_of_thoughts: ChainOfThoughts, messages: list[dict], user_message: str) -> str:
        """
        Processes the chain of thoughts by answering each step and composing the final answer.

        Args:
            chain_of_thoughts (ChainOfThoughts): The chain of thoughts to process.

        Returns:
            str: The final answer based on the chain of thoughts.
        """
        logger.info("(execute_chain_of_thoughts) Processing chain of thoughts.")
        step_messages = [
            ChatCompletionSystemMessageParam(
                content=(
                    "You have to solve this chain of thoughts in a straightforward way."
                ),
                role="system",
            )
        ] + messages
        for step in chain_of_thoughts.steps:
            logger.info(
                "(execute_chain_of_thoughts) Processing step: %s - Description: %s", step.description, step.title
            )
            step_content = self.answer_step(step, messages=step_messages)
            logger.debug
            logger.debug("Step content: %s", step_content)
            step_messages.append(
                ChatCompletionUserMessageParam(
                    content=f"Step: {step.title}\n Description: {step.description}",
                    role="user",
                )
            )
            step_messages.append(
                ChatCompletionAssistantMessageParam(
                    content=step_content,
                    role="assistant",
                )
            )

        step_messages.append(
            ChatCompletionUserMessageParam(
                content=(
                    f"User Input: {user_message}\n"
                    "Read all the messages calmly.\n"
                    "Provide a final answer listing all steps details and answers."
                    "Write a final report based on the context and the steps."
                ),
                role="user",
            )
        )

        chain_of_thoughts_response = self.client.chat.completions.create(
            messages=step_messages,
            model=self.valves.MODEL_NAME,
            temperature=0.15,
        )

        logger.debug(
            "(execute_chain_of_thoughts) Chain of thoughts response: %s",
            chain_of_thoughts_response.choices[0].message.content
        )
        self.total_of_chain_of_thoughts_executions += 1
        return chain_of_thoughts_response.choices[0].message.content

    def user_messages_answer(self, messages: List[dict], user_message: str) -> str:
        logger.info("(user_messages_answer) Processing user messages.")
        logger.debug(
            "(user_messages_answer) Messages: %s", messages
        )

        if self.total_of_chain_of_thoughts_executions == 0:
            current_messages = self.create_validation_message(messages)
        else:
            current_messages = self.create_chain_of_thoughts_messages(messages)

        chat_completion = self.client.chat.completions.create(
            messages=current_messages,
            model=self.valves.MODEL_NAME,
            temperature=0,
            tools=self.get_tools_for_chain_of_thoughts()
        )

        logger.debug(
            "(user_messages_answer) Chat completion: %s",
            chat_completion.choices[0].message.model_dump_json()
        )
        current_message = chat_completion.choices[0].message
        current_messages.append(current_message)

        if chat_completion.choices[0].finish_reason == "tool_calls":
            logger.info("(user_messages_answer) Handling tool calls.")
            return self.handle_tool_calls(current_message, current_messages, user_message)
        else:
            logger.info("(user_messages_answer) No tool calls detected.")
            logger.debug(
                "(user_messages_answer) Current content message: %s", current_message.model_dump_json()
            )
            return current_message.content

    def create_validation_message(self, messages: List[dict]) -> List[dict]:
        return messages + [
            ChatCompletionUserMessageParam(
                content=(
                    "Validate the user input based on the context."
                ),
                role="user",
            ),
        ]

    def create_chain_of_thoughts_messages(self, messages: List[dict]) -> List[dict]:
        return messages + [
            ChatCompletionUserMessageParam(
                content=(
                    "You have to solve this chain of thoughts in a straightforward way."
                    "Write only the necessary using few steps on the chain of thoughts."
                ),
                role="user",
            ),
        ]

    def get_tools_for_chain_of_thoughts(self) -> Union[None, List[ChatCompletionToolParam]]:
        logger.info(
            "(get_tools_for_chain_of_thoughts) Getting tools for chain of thoughts."
        )
        if self.total_of_chain_of_thoughts_executions <= self.valves.MAX_CHAIN_OF_THOUGHTS_EXECUTION:
            logger.info(
                "(get_tools_for_chain_of_thoughts) Tool is available for chain of thoughts."
            )
            chain_of_thoughts_schema = ChainOfThoughts.model_json_schema()
            chain_of_thoughts_schema["additionalProperties"] = False
            return [
                ChatCompletionToolParam(
                    function=FunctionDefinition(
                        name="chain_of_thoughts",
                        description="Used to plan a chain of thoughts to answer the user.",
                        parameters=chain_of_thoughts_schema,
                        strict=True,
                    ),
                    type="function",
                )
            ]

        logger.info(
            "(get_tools_for_chain_of_thoughts) No tool available for chain of thoughts."
        )
        return None

    def handle_tool_calls(self, current_message, current_messages, user_message) -> str:
        logger.info("(handle_tool_calls) Handling tool calls.")
        for tool_call in current_message.tool_calls:
            previous_messages = current_messages[:-1]
            if tool_call.function.name == "chain_of_thoughts":
                logger.info("(handle_tool_calls) Handling chain of thoughts.")
                chain_of_thoughts = ChainOfThoughts(
                    **json.loads(tool_call.function.arguments)
                )
                logger.debug(
                    "(handle_tool_calls) Chain of thoughts: %s",
                    chain_of_thoughts.model_dump_json()
                )

                chain_of_thoughts_answer = self.execute_chain_of_thoughts(
                    chain_of_thoughts,
                    messages=previous_messages,
                    user_message=user_message
                )

                logger.info(
                    "(handle_tool_calls) Chain of thoughts answer: %s",
                    chain_of_thoughts_answer
                )

                current_messages.append(
                    ChatCompletionToolMessageParam(
                        tool_call_id=tool_call.id,
                        content=chain_of_thoughts_answer,
                        role="tool"
                    )
                )
            elif tool_call.function.name == "code_execution":
                try:
                    logger.info("(handle_tool_calls) Handling code execution.")
                    json_args = json.loads(tool_call.function.arguments)
                    code = json_args.get("code")
                    logger.debug(
                        "(handle_tool_calls) Code to execute:\n%s", code)
                    code_execution_result, code_execution_return = self.execute_python_code(
                        code)

                    if code_execution_return != 0:
                        error_message = f"Error: {code_execution_result}"
                        logger.error(
                            "(handle_tool_calls) Code execution failed: %s",
                            error_message
                        )
                        current_messages.append(
                            ChatCompletionToolMessageParam(
                                tool_call_id=tool_call.id,
                                content=error_message,
                                role="tool",
                            )
                        )
                        continue

                    current_messages.append(
                        ChatCompletionToolMessageParam(
                            tool_call_id=tool_call.id,
                            content=code_execution_result,
                            role="tool",
                        )
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        "(handle_tool_calls) JSON decode error: %s",
                        e
                    )
                    continue

        answer = self.craft_answer(
            messages=current_messages,
            user_message=user_message
        )

        current_messages.append(
            ChatCompletionAssistantMessageParam(
                content=answer,
                role="assistant"
            )
        )

        return self.user_messages_answer(current_messages, user_message)

    def craft_answer(self, messages: list[dict], user_message) -> dict:
        logger.info("(craft_answer) Crafting the final answer.")
        messages = messages + [
            ChatCompletionUserMessageParam(
                content=(
                    f"Original User Input: {user_message}\n"
                    "Read all the messages calmly.\n"
                    "Provide the final answer to the user based on the context and the steps."
                    "Do not change the context, do not hallucinate, and do not add information that was not requested."
                    "Write your answer based on the context and facts on the messages."
                ),
                role="user",
            )
        ]

        answer = self.client.chat.completions.create(
            messages=messages,
            model=self.valves.MODEL_NAME,
            temperature=0.7,
        )
        logger.debug(
            "(craft_answer) Final answer: %s", answer.choices[0].message.content
        )
        return answer.choices[0].message.content

    def generate_title(self, user_message: str) -> str:
        logger.info("(generate_title) Generating a title.")
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            model="gpt-4o-mini",
        )
        return chat_completion.choices[0].message.content

    def create_final_answer(self, user_message: str, messages: list[dict]):
        messages = messages + [
            ChatCompletionUserMessageParam(
                content=(
                    f"Given this original task {user_message}\n\n\n"
                    "Read all the messages calmly.\n"
                    "Answer the user based on the context and the steps."
                ),
                role="user",
            ),
        ]
        logger.info("(create_final_answer) Creating the final answer.")
        final_answer_response = self.client.chat.completions.create(
            messages=messages,
            model=self.valves.MODEL_NAME,
            temperature=0.7,
        )

        logger.debug(
            "(create_final_answer) Final answer response: %s",
            final_answer_response.choices[0].message.content
        )

        return final_answer_response

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.info("(pipe) Processing user message.")
        MODEL_NAME = self.valves.MODEL_NAME

        if self.is_title_generation:
            return self.generate_title(user_message)

        first_messages = [
            ChatCompletionSystemMessageParam(
                content="First answer the user input based on the context.",
                role="system",
            )
        ] + messages

        first_answer = self.client.chat.completions.create(
            messages=first_messages,
            model=MODEL_NAME,
            temperature=0.7,
        )
        messages.append(first_answer.choices[0].message)
        logger.debug(
            "(pipe) First answer: %s",
            first_answer.choices[0].message
        )

        user_messages_answer_response = self.user_messages_answer(
            messages, user_message
        )

        messages.append(
            ChatCompletionAssistantMessageParam(
                content=user_messages_answer_response,
                role="assistant",
            )
        )

        final_answer = self.create_final_answer(
            user_message, messages
        )

        return final_answer.choices[0].message.content
