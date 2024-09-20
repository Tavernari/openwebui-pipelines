"""
Title: o1-reasoning Pipeline powered by OpenAI
Author: Victor Carvalho Tavernari
Date: 2024-09-18
Version: 0.2-alpha
License: MIT
Description:
    This pipeline processes user messages and interacts with the OpenAI API.
    It breaks down user input into steps, processes each step, and composes a final answer based on the steps and their results.
Requirements: openai, pydantic, duckduckgo_search, beautifulsoup4
"""

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from enum import Enum
from typing import Iterable, List, Optional
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
from duckduckgo_search import DDGS
import urllib
from bs4 import BeautifulSoup


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
logger.setLevel(logging.INFO)

for handler in logger.handlers:
    logger.removeHandler(handler)

# Ensure we don't add duplicate handlers
if not logger.handlers:
    logger.addHandler(console_handler)


# MARK: Models

# Model for the 'user_input' API

class UserInputType(str, Enum):
    logical_resolution = 'logical_resolution'
    code = 'code'
    article_generation = 'article_generation'
    research = 'research'
    history = 'history'
    biology = 'biology'
    physics = 'physics'
    chemistry = 'chemistry'
    mathematics = 'mathematics'
    geography = 'geography'
    literature = 'literature'
    philosophy = 'philosophy'
    psychology = 'psychology'
    sociology = 'sociology'
    economics = 'economics'
    politics = 'politics'
    law = 'law'
    computer_science = 'computer_science'
    engineering = 'engineering'
    medicine = 'medicine'
    generic = 'generic'


class UserInputDetails(BaseModel):
    """
    Model representing the user language.
    """

    language: str = Field(
        ...,
        description="The language of the user in iso format.",
    )

    original_question: str = Field(
        ...,
        description="The original question",
    )

    what_user_wants: str = Field(
        ...,
        description="Explain detailed what the input wants to know or perform.",
    )

    fast_answer: str = Field(
        ...,
        description=(
            "Based on the request write a fast answer."
            "Even a fast answer must be based on the reasoning."
            "Never create information that you don't have a high confidence."
            "If you don't have the information, you must write that you don't have the information."
            "If it is related numbers, statistics, etc, you should not write the fast answer."
            "It only accept string human readable format."
        ),
    )

    fast_answer_reasoning: list[str] = Field(
        ...,
        description="A list of reasons to support the fast answer.",
    )

    type: str = Field(
        ...,
        description=(
            "The type of the user input."
            f"Possible values: {', '.join(UserInputType.__members__.keys())}"
        ),
    )

    model_config = ConfigDict(
        extra='forbid',
        strict=True,
    )


# Model for the 'text' search API

class TextWebSearchRequest(BaseModel):
    """
    Model representing a text web search request for DuckDuckGo.
    """
    keywords: str = Field(
        ...,
        description="Keywords for the query."
    )
    region: str = Field(
        description="Region code, e.g., 'wt-wt', 'us-en', 'uk-en', 'ru-ru'. Defaults to 'wt-wt'."
    )
    safesearch: str = Field(
        description="Safe search setting: 'on', 'moderate', 'off'. Defaults to 'moderate'."
    )
    timelimit: Optional[str] = Field(
        description="Time limit for search results: 'd' (day), 'w' (week), 'm' (month), 'y' (year). Defaults to None."
    )
    backend: str = Field(
        description=(
            "Backend to use for the search: 'api', 'html', 'lite'. Defaults to 'api'.\n"
            " - 'api' collects data from https://duckduckgo.com\n"
            " - 'html' collects data from https://html.duckduckgo.com\n"
            " - 'lite' collects data from https://lite.duckduckgo.com"
        )
    )
    max_results: Optional[int] = Field(
        description="Maximum number of results to return. If None, returns results only from the first response."
    )

    model_config = ConfigDict(
        extra='forbid',
    )

# Model for get content from some html URL


class WebSiteContent(BaseModel):
    """
    Model representing a website content.
    """
    url: str = Field(
        ...,
        description="URL of the website."
    )

    model_config = ConfigDict(
        extra='forbid',
    )

# Model for the 'answers' API


class WebAnswersRequest(BaseModel):
    """
    Model representing an instant answers request for DuckDuckGo.
    """
    keywords: str = Field(
        ...,
        description="Keywords for the query."
    )

    model_config = ConfigDict(
        extra='forbid',
    )

# Model for the 'news' search API


class NewsSearchRequest(BaseModel):
    """
    Model representing a news search request for DuckDuckGo.
    """
    keywords: str = Field(
        ...,
        description="Keywords for the query."
    )
    region: str = Field(
        description="Region code, e.g., 'wt-wt', 'us-en', 'uk-en', 'ru-ru'. Defaults to 'wt-wt'."
    )
    safesearch: str = Field(
        description="Safe search setting: 'on', 'moderate', 'off'. Defaults to 'moderate'."
    )
    timelimit: Optional[str] = Field(
        description="Time limit for search results: 'd' (day), 'w' (week), 'm' (month). Defaults to None."
    )
    max_results: Optional[int] = Field(
        description="Maximum number of results to return. If None, returns results only from the first response."
    )

    model_config = ConfigDict(
        extra='forbid',
    )

# Model for the 'maps' search API


class MapsSearchRequest(BaseModel):
    """
    Model representing a maps search request for DuckDuckGo.
    """
    keywords: str = Field(
        ...,
        description="Keywords for the query."
    )
    place: Optional[str] = Field(
        description="Specific place to search. If set, other location parameters are ignored."
    )
    street: Optional[str] = Field(
        description="Street address for the search."
    )
    city: Optional[str] = Field(
        description="City for the search."
    )
    county: Optional[str] = Field(
        description="County for the search."
    )
    state: Optional[str] = Field(
        description="State for the search."
    )
    country: Optional[str] = Field(
        description="Country for the search."
    )
    postalcode: Optional[str] = Field(
        description="Postal code for the search."
    )
    latitude: Optional[str] = Field(
        description=(
            "Latitude coordinate for the search. If both latitude and longitude are set, "
            "other location parameters are ignored."
        )
    )
    longitude: Optional[str] = Field(
        description="Longitude coordinate for the search."
    )
    radius: int = Field(
        description="Radius to expand the search area in kilometers. Defaults to 0."
    )
    max_results: Optional[int] = Field(
        description="Maximum number of results to return. If None, returns results only from the first response."
    )

    model_config = ConfigDict(
        extra='forbid',
    )


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

    # MARK: Tools

    def web_site_content_tool(self) -> ChatCompletionToolParam:
        web_site_content_schema = WebSiteContent.model_json_schema()
        web_site_content_schema["additionalProperties"] = False
        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name="web_site_content",
                description="Get the content of a website.",
                parameters=web_site_content_schema,
                strict=True,
            ),
            type="function",
        )

    def news_tool(self) -> ChatCompletionToolParam:
        news_schema = NewsSearchRequest.model_json_schema()
        news_schema["additionalProperties"] = False
        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name="news",
                description="Search for news related to the context.",
                parameters=news_schema,
                strict=True,
            ),
            type="function",
        )

    def web_answers_tool(self) -> ChatCompletionToolParam:
        web_answers_schema = WebAnswersRequest.model_json_schema()
        web_answers_schema["additionalProperties"] = False
        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name="web_answers",
                description="Search for instant answers related to the context.",
                parameters=web_answers_schema,
                strict=True,
            ),
            type="function",
        )

    def text_web_search_tool(self) -> ChatCompletionToolParam:
        text_web_search_schema = TextWebSearchRequest.model_json_schema()
        text_web_search_schema["additionalProperties"] = False
        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name="text_web_search",
                description="Search the web for text related to the context.",
                parameters=text_web_search_schema,
                strict=True,
            ),
            type="function",
        )

    def maps_search_tool(self) -> ChatCompletionToolParam:
        maps_search_schema = MapsSearchRequest.model_json_schema()
        maps_search_schema["additionalProperties"] = False
        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name="maps_search",
                description="Search for maps related to the context.",
                parameters=maps_search_schema,
                strict=True,
            ),
            type="function",
        )

    def chain_of_thoughts_tool(self) -> ChatCompletionToolParam:
        chain_of_thoughts_schema = ChainOfThoughts.model_json_schema()
        chain_of_thoughts_schema["additionalProperties"] = False
        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name="chain_of_thoughts",
                description=(
                    "Process the chain of thoughts to help to answer the step."
                    "This toos is to realize how to solve the problem using a chain of thoughts."
                    "The steps represent small piece of thoughts to solve the problem."
                ),
                parameters=chain_of_thoughts_schema,
                strict=True,
            ),
            type="function",
        )

    def code_execution_tool(self) -> ChatCompletionToolParam:
        code_execution_schema = CodeExecution.model_json_schema()
        code_execution_schema["additionalProperties"] = False
        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name="code_execution",
                description=(
                    "Execute the Python code to check the result."
                    "The code must be written to solve or validate some sub-problem."
                    "The result must be printed using the print statement."
                    "Write print to show no only the final result but to show the steps."
                ),
                parameters=code_execution_schema,
                strict=True,
            ),
            type="function",
        )

# MARK: Tool Execution

    def execute_web_site_content_tool(self, web_site_content: WebSiteContent) -> str:
        """
        Executes the web site content tool with the given parameters.
        """
        logger.info(
            "(execute_web_site_content_tool) Getting content from URL: %s", web_site_content.url
        )
        try:
            html = urllib.urlopen(web_site_content.url).read()
            soup = BeautifulSoup(html)

            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # rip it out

            # get text
            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip()
                      for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            logger.debug(
                "(execute_web_site_content_tool) Web site content: %s", text
            )
            logger.info(
                "(execute_web_site_content_tool) Web site content complete for URL: %s", web_site_content.url
            )
            return text
        except Exception as e:
            logger.exception(
                "(execute_web_site_content_tool) Web site content failed for URL: %s", web_site_content.url
            )
            return str(e)

    def execute_python_code(self, code: CodeExecution) -> (str, int):  # type: ignore
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
                ["python", "-c", code.code],
                capture_output=True,
                text=True,
                check=True,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            logger.debug(
                "(execute_python_code) Code execution result: %s - %s",
                stdout, stderr
            )

            # if fails, return the error message
            if result.returncode != 0:
                return stderr

            return stdout
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip()
            logger.error(
                "(execute_python_code) Code execution failed: %s",
                stderr
            )

            return stderr
        except Exception as e:
            logger.exception("(execute_python_code) Code execution failed.")
            return str(e)

    def execute_text_web_search_tool(self, text_web_search: TextWebSearchRequest) -> str:
        """
        Executes the text web search tool with the given parameters.
        """
        logger.info(
            "(execute_text_web_search_tool) Searching for text: %s", text_web_search.keywords
        )
        results = DDGS().text(
            keywords=text_web_search.keywords,
            region=text_web_search.region,
            safesearch=text_web_search.safesearch,
            timelimit=text_web_search.timelimit,
            backend=text_web_search.backend,
            max_results=text_web_search.max_results,
        )

        result = json.dumps(results, indent=0)
        logger.debug(
            "(execute_text_web_search_tool) Text search results: %s", result
        )
        logger.info(
            "(execute_text_web_search_tool) Text search complete for keywords: %s", text_web_search.keywords
        )
        return result

    def execute_web_answers_tool(self, web_answers: WebAnswersRequest) -> str:
        """
        Executes the web answers tool with the given parameters.
        """
        logger.info(
            "(execute_web_answers_tool) Searching for web answers: %s", web_answers.keywords
        )
        results = DDGS().answers(
            keywords=web_answers.keywords,
        )

        result = json.dumps(results, indent=0)
        logger.debug(
            "(execute_web_answers_tool) Web answers results: %s", result
        )
        logger.info(
            "(execute_web_answers_tool) Web answers search complete for keywords: %s", web_answers.keywords
        )
        return result

    def execute_maps_search_tool(self, maps_search: MapsSearchRequest) -> str:
        """
        Executes the maps search tool with the given parameters.
        """
        logger.info(
            "(execute_maps_search_tool) Searching for maps: %s", maps_search.keywords
        )
        results = DDGS().maps(
            keywords=maps_search.keywords,
            place=maps_search.place,
            street=maps_search.street,
            city=maps_search.city,
            county=maps_search.county,
            state=maps_search.state,
            country=maps_search.country,
            postalcode=maps_search.postalcode,
            latitude=maps_search.latitude,
            longitude=maps_search.longitude,
            radius=maps_search.radius,
            max_results=maps_search.max_results,
        )

        result = json.dumps(results, indent=0)
        logger.debug(
            "(execute_maps_search_tool) Maps search results: %s", result
        )
        logger.info(
            "(execute_maps_search_tool) Maps search complete for keywords: %s", maps_search.keywords
        )
        return result

    def execute_news_tool(self, news: NewsSearchRequest) -> str:
        """
        Executes the news search tool with the given parameters.
        """

        logger.info(
            "(execute_news_tool) Searching for news: %s", news.keywords
        )
        results = DDGS().news(
            keywords=news.keywords,
            region=news.region,
            safesearch=news.safesearch,
            timelimit=news.timelimit,
            max_results=news.max_results,
        )

        result = json.dumps(results, indent=0)
        logger.debug(
            "(execute_news_tool) News search results: %s", result
        )
        logger.info(
            "(execute_news_tool) News search complete for keywords: %s", news.keywords
        )
        return result

    # MARK: Tool Routing

    def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        switcher = {
            "news": (self.execute_news_tool, NewsSearchRequest),
            "web_answers": (self.execute_web_answers_tool, WebAnswersRequest),
            "text_web_search": (self.execute_text_web_search_tool, TextWebSearchRequest),
            "maps_search": (self.execute_maps_search_tool, MapsSearchRequest),
            "code_execution": (self.execute_python_code, CodeExecution),
            "web_site_content": (self.execute_web_site_content_tool, WebSiteContent),
        }

        tool_function, tool_model = switcher.get(tool_name, (None, None))

        if tool_function is None:
            return "Tool not found."

        tool_args = tool_model(**tool_args)
        return tool_function(tool_args)

    def available_tools(self) -> List[ChatCompletionToolParam]:
        return [
            self.news_tool(),
            self.web_answers_tool(),
            self.text_web_search_tool(),
            self.maps_search_tool(),
            self.code_execution_tool(),
            self.web_site_content_tool(),
        ]

    # MARK: Chain of Thoughts Execution

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

        messages = messages + [
            ChatCompletionUserMessageParam(
                content=(
                    f"Step: {step.title}"
                    f"Description: {step.description}"
                    "You must follow the step proposal to provide your answer."
                    "Only if it is possible to represent it as code, you can check it using python code, so write the code and execute it using the code_execution tool."
                    f"IMPORTANT: You must write it using the same language as the user language: {self.user_input_details.language}."
                ),
                role="user",
            ),
        ]

        logger.debug("(answer_step) messages: %s", messages)

        response = self.call_text_gpt(
            messages=messages,
            tools=self.available_tools(),
        )

        message = response.choices[0].message
        logger.debug(
            "(answer_step) message: %s", message.content
        )
        messages.append(message)

        if response.choices[0].finish_reason == "tool_calls":
            for tool_call in response.choices[0].message.tool_calls:
                tool_response = self.execute_tool(
                    tool_call.function.name,
                    json.loads(
                        tool_call.function.arguments
                    )
                )

                logger.debug(
                    "(answer_step) Tool response: %s", tool_response
                )

                messages.append(
                    ChatCompletionToolMessageParam(
                        tool_call_id=tool_call.id,
                        content=tool_response,
                        role="tool",
                    )
                )

            messages.append(
                ChatCompletionUserMessageParam(
                    content=(
                        "You must write the final answer based on the context and the steps."
                        "Write each step details and answers."
                        "Write a final report based on the context and the steps."
                        "You must present the source of the information, like the link of the news, the code execution result, etc."
                        f"IMPORTANT: You must write it using the same language as the user language: {self.user_input_details.language}."
                    ),
                    role="user",
                ),
            )

            response = self.call_text_gpt(messages=messages)
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
                    content=(
                        f"Step: {step.title}\n Description: {step.description}"
                    ),
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
                    "Read all the messages calmly.\n"
                    "Provide a final answer listing all steps details and answers."
                    "Write a final report based on the context and the steps."
                    "Always present the source of the information, like the link of the source, the code execution result, etc."
                    "IMPORTANT:\n"
                    f"- You must write it using the same language as the user language: {self.user_input_details.language}."
                    f"- The user wants to know: {self.user_input_details.what_user_wants}."
                    f"- The original question is: {self.user_input_details.original_question}."

                ),
                role="user",
            )
        )

        chain_of_thoughts_response = self.call_text_gpt(messages=step_messages)

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

        tools = None
        if self.total_of_chain_of_thoughts_executions < self.valves.MAX_CHAIN_OF_THOUGHTS_EXECUTION:
            logger.info(
                "(user_messages_answer) Executing chain of thoughts."
            )
            current_messages = self.create_chain_of_thoughts_messages(messages)
            tools = [self.chain_of_thoughts_tool()]

            logger.debug(
                "(user_messages_answer) Current messages: %s", current_messages
            )
            chat_completion = self.call_text_gpt(
                messages=current_messages,
                tools=tools,
                temperature=0.5
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
        else:
            return messages[-1].get("content", "")

    # MARK: Prompts Definition

    def make_prompt_for_logical_resolution(self) -> str:
        return (
            f"You are an expert in logical reasoning.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Original Question: {self.user_input_details.original_question}\n"
            f"- What the user wants: {self.user_input_details.what_user_wants}\n"
            f"- Preliminary Answer: {self.user_input_details.fast_answer}\n"
            f"- Preliminary Reasoning: {', '.join(self.user_input_details.fast_answer_reasoning)}\n\n"
            "Instructions:\n"
            "- Solve the problem step-by-step using clear and concise logical reasoning.\n"
            "- Break down the problem into smaller components.\n"
            "- Provide justification for each step of your reasoning.\n"
            "- Ensure that your final conclusion addresses the user's original question.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_code(self) -> str:
        return (
            f"You are a seasoned software engineer proficient in multiple programming languages.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Original Question: {self.user_input_details.original_question}\n"
            f"- What the user wants: {self.user_input_details.what_user_wants}\n"
            f"- Preliminary Answer: {self.user_input_details.fast_answer}\n"
            f"- Preliminary Reasoning: {', '.join(self.user_input_details.fast_answer_reasoning)}\n\n"
            "Instructions:\n"
            "- Provide a code solution that fulfills the user's requirements.\n"
            "- Explain your code thoroughly, commenting on each significant part.\n"
            "- Ensure the code follows best practices and is optimized for efficiency.\n"
            "- If necessary, include sample inputs and outputs to demonstrate functionality.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_article_generation(self) -> str:
        return (
            f"You are an experienced writer tasked with generating an informative article.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Write a comprehensive article covering the topic in detail.\n"
            "- Structure the article with an introduction, body, and conclusion.\n"
            "- Use clear headings and subheadings to organize content.\n"
            "- Incorporate relevant data and examples where appropriate.\n"
            "- Ensure the content is original and free of plagiarism.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_research(self) -> str:
        return (
            f"You are a diligent researcher.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Research Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Conduct thorough research on the topic.\n"
            "- Summarize your findings in a detailed report.\n"
            "- Include data, statistics, and cite credible sources.\n"
            "- Provide links to sources where applicable.\n"
            "- Remain objective and present multiple perspectives if relevant.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_history(self) -> str:
        return (
            f"You are a knowledgeable historian.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Historical Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Provide a detailed explanation of the historical events or context.\n"
            "- Include relevant dates, figures, and significant occurrences.\n"
            "- Use credible sources and, if possible, provide references.\n"
            "- Ensure your response is accurate and informative.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_biology(self) -> str:
        return (
            f"You are an expert biologist.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Biological Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Explain the biological concepts related to the topic.\n"
            "- Use clear and precise scientific terminology.\n"
            "- Provide examples or diagrams if helpful.\n"
            "- Ensure your explanation is accurate and up-to-date.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_physics(self) -> str:
        return (
            f"You are a physicist with deep knowledge in the field.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Physics Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Explain the physical principles or theories involved.\n"
            "- Use equations and formulas if necessary, providing explanations.\n"
            "- Provide real-world examples to illustrate concepts.\n"
            "- Ensure accuracy and clarity in your explanation.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_chemistry(self) -> str:
        return (
            f"You are a chemist with extensive knowledge.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Chemistry Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Explain the chemical concepts or reactions involved.\n"
            "- Use chemical equations if appropriate, with explanations.\n"
            "- Provide examples or applications in real life.\n"
            "- Ensure your explanation is accurate and detailed.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_mathematics(self) -> str:
        return (
            f"You are a mathematician skilled in explaining mathematical concepts.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Mathematical Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Provide a step-by-step explanation of the mathematical problem or concept.\n"
            "- Include formulas, theorems, or proofs as necessary.\n"
            "- Use examples to illustrate key points.\n"
            "- Ensure clarity and accuracy in your explanation.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_geography(self) -> str:
        return (
            f"You are a geographer with extensive knowledge of physical and human geography.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Geography Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Provide detailed information on the geographical topic.\n"
            "- Include data on location, climate, demographics, or other relevant factors.\n"
            "- Use maps or diagrams if helpful.\n"
            "- Ensure accuracy and thoroughness in your response.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_literature(self) -> str:
        return (
            f"You are a literary scholar well-versed in literature.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Literature Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Analyze the literary work or concept in detail.\n"
            "- Discuss themes, characters, and stylistic elements.\n"
            "- Provide critical interpretations or contextual information.\n"
            "- Ensure your analysis is insightful and well-supported.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_philosophy(self) -> str:
        return (
            f"You are a philosopher with deep understanding of philosophical concepts.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Philosophical Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Explain the philosophical ideas or arguments involved.\n"
            "- Discuss different viewpoints or schools of thought.\n"
            "- Provide examples or thought experiments where appropriate.\n"
            "- Ensure clarity and depth in your explanation.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_psychology(self) -> str:
        return (
            f"You are a psychologist knowledgeable in psychological theories and practices.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Psychology Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Explain the psychological concepts or theories involved.\n"
            "- Use real-world examples or case studies if helpful.\n"
            "- Discuss implications or applications in everyday life.\n"
            "- Ensure your explanation is accurate and empathetic.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_sociology(self) -> str:
        return (
            f"You are a sociologist with expertise in social dynamics and structures.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Sociology Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Analyze the social phenomena or concepts involved.\n"
            "- Provide statistical data or studies if applicable.\n"
            "- Discuss theories and perspectives relevant to the topic.\n"
            "- Ensure your explanation is thorough and insightful.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_economics(self) -> str:
        return (
            f"You are an economist with a deep understanding of economic principles.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Economics Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Explain the economic concepts or models involved.\n"
            "- Use data, graphs, or equations where appropriate.\n"
            "- Discuss real-world applications or implications.\n"
            "- Ensure clarity and accuracy in your explanation.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_politics(self) -> str:
        return (
            f"You are a political scientist knowledgeable about political systems and theories.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Political Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Provide an analysis of the political issue or concept.\n"
            "- Include historical context or current events if relevant.\n"
            "- Discuss different perspectives or ideologies.\n"
            "- Ensure objectivity and depth in your explanation.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_law(self) -> str:
        return (
            f"You are a legal expert knowledgeable in laws and regulations.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Legal Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Explain the legal principles or statutes involved.\n"
            "- Provide case examples or precedents if applicable.\n"
            "- Ensure that your explanation is accurate and clear.\n"
            "- Note: Do not provide legal advice; focus on information.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_computer_science(self) -> str:
        return (
            f"You are a computer scientist with expertise in computer science concepts.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Computer Science Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Explain the computer science concepts or algorithms involved.\n"
            "- Use code snippets or diagrams if helpful.\n"
            "- Provide examples of applications or implementations.\n"
            "- Ensure clarity and accuracy in your explanation.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_engineering(self) -> str:
        return (
            f"You are an engineer with expertise in engineering principles.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Engineering Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Explain the engineering concepts or processes involved.\n"
            "- Use diagrams, formulas, or calculations where appropriate.\n"
            "- Provide examples of real-world applications.\n"
            "- Ensure precision and clarity in your explanation.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_medicine(self) -> str:
        return (
            f"You are a medical professional knowledgeable in medical science.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Medical Topic: {self.user_input_details.what_user_wants}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Explain the medical concepts or conditions involved.\n"
            "- Use medical terminology appropriately, with explanations.\n"
            "- Provide information on symptoms, treatments, or mechanisms.\n"
            "- Ensure your explanation is accurate and compassionate.\n"
            "- Note: This is for informational purposes only; do not provide medical advice.\n"
            "- Write your response in the same language as the user.\n"
        )

    def make_prompt_for_generic(self) -> str:
        return (
            f"You are a knowledgeable assistant.\n\n"
            f"User Input Details:\n"
            f"- Language: {self.user_input_details.language}\n"
            f"- Original Question: {self.user_input_details.original_question}\n\n"
            "Instructions:\n"
            "- Provide a clear and accurate response to the user's question.\n"
            "- Ensure your answer is comprehensive and addresses all aspects of the query.\n"
            "- Use simple and understandable language.\n"
            "- Write your response in the same language as the user.\n"
        )

    # MARK: Chain of Thoughts Prompt Handler

    def create_chain_of_thoughts_messages(self, messages: List[dict]) -> List[dict]:
        prompt_function = {
            UserInputType.logical_resolution: self.make_prompt_for_logical_resolution,
            UserInputType.code: self.make_prompt_for_code,
            UserInputType.article_generation: self.make_prompt_for_article_generation,
            UserInputType.research: self.make_prompt_for_research,
            UserInputType.history: self.make_prompt_for_history,
            UserInputType.biology: self.make_prompt_for_biology,
            UserInputType.physics: self.make_prompt_for_physics,
            UserInputType.chemistry: self.make_prompt_for_chemistry,
            UserInputType.mathematics: self.make_prompt_for_mathematics,
            UserInputType.geography: self.make_prompt_for_geography,
            UserInputType.literature: self.make_prompt_for_literature,
            UserInputType.philosophy: self.make_prompt_for_philosophy,
            UserInputType.psychology: self.make_prompt_for_psychology,
            UserInputType.sociology: self.make_prompt_for_sociology,
            UserInputType.economics: self.make_prompt_for_economics,
            UserInputType.politics: self.make_prompt_for_politics,
            UserInputType.law: self.make_prompt_for_law,
            UserInputType.computer_science: self.make_prompt_for_computer_science,
            UserInputType.engineering: self.make_prompt_for_engineering,
            UserInputType.medicine: self.make_prompt_for_medicine,
            UserInputType.generic: self.make_prompt_for_generic,
        }

        prompt_func = prompt_function.get(
            self.user_input_details.type if self.user_input_details else "generic",
            self.make_prompt_for_generic
        )
        prompt = prompt_func()

        return messages + [
            {
                "role": "user",
                "content": (
                    f"{prompt}\n"
                    "Additional Instructions:\n"
                    f"- Today is {datetime.now().strftime('%Y-%m-%d')}.\n"
                    "- Solve this task using a straightforward chain of thought.\n"
                    "- The steps should represent small pieces of thought to solve the problem.\n"
                    "- Use tools like code execution, searching for news, text, maps, or loading website content as needed.\n"
                    "- Make a linear sequence of thoughts.\n"
                    "- Even simple steps can be useful.\n"
                    "- If the task involves a specific point in time, base your reasoning on the date mentioned.\n"
                    "- Try to validate your preliminary answer and reasoning through these steps.\n"
                    "- Provide sources for information such as links to news articles or code snippets.\n"
                    "- IMPORTANT: Attempt to challenge your initial theory with proof steps and reasoning.\n"
                ),
            },
        ]

    def handle_tool_calls(self, current_message, current_messages, user_message) -> str:
        logger.info("(handle_tool_calls) Handling tool calls.")
        for tool_call in current_message.tool_calls:
            previous_messages = current_messages[:-1]
            if tool_call.function.name == "chain_of_thoughts":
                logger.info("(handle_tool_calls) Handling chain of thoughts.")
                chain_of_thoughts = ChainOfThoughts(
                    **json.loads(tool_call.function.arguments)
                )

                for step in chain_of_thoughts.steps:
                    logger.info(
                        "(handle_tool_calls) Step: %s - Description: %s",
                        step.title, step.description
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

        answer = self.create_answer(
            messages=current_messages,
            user_message=user_message
        )

        logger.debug(
            "(handle_tool_calls) Answer: %s", answer
        )

        current_messages.append(
            ChatCompletionAssistantMessageParam(
                content=answer,
                role="assistant"
            )
        )

        return self.user_messages_answer(current_messages, user_message)

    def generate_title(self, user_message: str) -> str:
        logger.info("(generate_title) Generating a title.")
        chat_completion = self.call_text_gpt(
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            temperature=0.7
        )
        return chat_completion.choices[0].message.content

    def create_answer(self, user_message, messages: list[dict]):
        logger.info("(create_answer) Creating the answer.")
        messages = messages + [
            ChatCompletionUserMessageParam(
                content=(
                    "Write the final answer based on the context."
                    "Your final answer must explain the steps and the reasoning."
                    "Then write a final answer based on the context and the steps."
                    "You must present the source of the information, like the link of the news, the code execution result, etc."
                    "IMPORTANT:\n"
                    f"- You must write it using the same language as the user language: {self.user_input_details.language}."
                    f"- The user wants to know: {self.user_input_details.what_user_wants}."
                    f"- The original question is: {self.user_input_details.original_question}."
                ),
                role="user",
            ),
        ]
        final_answer_response = self.call_text_gpt(
            messages=messages,
            temperature=0
        )

        answer = final_answer_response.choices[0].message.content

        logger.debug(
            "(create_answer) Final answer: %s",
            answer
        )

        return answer

    def extract_user_input_details(self, user_message: str) -> UserInputDetails:
        """
        Extracts the language of the user input.

        Args:
            user_message (str): The user input message.

        Returns:
            UserInputDetails: The language of the user input.
        """
        logger.info(
            "(extract_user_input_details) Extracting user input details."
        )

        json_schema = UserInputDetails.model_json_schema()

        # get all properties from the schema
        properties = json_schema.get("properties", {})

        # {'language': {'description': 'The language of the user in iso format.', 'title': 'Language', 'type': 'string'}, 'original_question': {'description': 'The original question', 'title': 'Original Question', 'type': 'string'}, 'what_user_wants': {'description': 'Explain detailed what the input wants to know or perform.', 'title': 'What User Wants', 'type': 'string'}, 'fast_answer': {'description': 'Based on the request write a fast answer.', 'title': 'Fast Answer', 'type': 'string'}, 'fast_answer_reasoning': {'description': 'A list of reasons to support the fast answer.', 'items': {'type': 'string'}, 'title': 'Fast Answer Reasoning', 'type': 'array'}}
        # write a properties description with this formmart
        # - key: type | description: description

        print(properties)

        properties_description = "\n".join(
            [
                f"- key: {key} | type: {value['type']} | description: {value['description']}"
                for key, value in properties.items()
            ]
        )

        print(properties)

        messages = [
            ChatCompletionUserMessageParam(
                content=(
                    f"Given the user input: {user_message}"
                    "Extract user input details."
                    "Your output must be in JSON\n"
                    "With these fields:\n"
                    f"{properties_description}"

                ),
                role="user",
            ),
        ]

        content = self.call_json_gpt(messages=messages)

        logger.info(
            "(extract_user_input_details) Response: %s",
            json.dumps(content, indent=2)
        )

        return UserInputDetails(**content)

    def call_json_gpt(self, messages: List[dict], temperature=0) -> dict | None:
        try:
            logger.info("(call_json_gpt) Calling JSON GPT.")
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.valves.MODEL_NAME,
                temperature=temperature,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            logger.debug(
                "(call_json_gpt) Response: %s", content
            )

            return json.loads(content)
        except Exception as e:
            logger.exception(
                f"(call_json_gpt) Error calling JSON GPT. {str(e)}"
            )
            return None

    def call_text_gpt(self, messages: List[dict], temperature=0.2, tools: Iterable[ChatCompletionToolParam] | None = None) -> str:
        try:
            logger.info("(call_text_gpt) Calling text GPT.")
            return self.client.chat.completions.create(
                messages=messages,
                model=self.valves.MODEL_NAME,
                temperature=temperature,
                tools=tools,
            )
        except Exception as e:
            logger.exception(
                "(call_text_gpt) Error calling text GPT."
            )
            return f"Error calling text GPT: {str(e)}"
        # MARK: Pipe execution

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logger.info("(pipe) Processing user message.")
        MODEL_NAME = self.valves.MODEL_NAME

        if self.is_title_generation:
            return self.generate_title(user_message)

        self.user_input_details = self.extract_user_input_details(user_message)

        logger.info(
            "(pipe) Starting chain of thoughts."
        )
        user_messages_answer_response = self.user_messages_answer(
            messages, user_message
        )
        logger.info(
            "(pipe) User messages answer response: %s",
            user_messages_answer_response
        )

        return user_messages_answer_response
