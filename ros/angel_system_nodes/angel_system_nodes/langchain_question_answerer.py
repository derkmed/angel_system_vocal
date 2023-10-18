import json
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import openai
import os
import queue
import rclpy
from rclpy.node import Node
import requests
from termcolor import colored
import threading

from angel_msgs.msg import InterpretedAudioUserEmotion, SystemTextResponse
from angel_utils import declare_and_get_parameters

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

IN_EMOTION_TOPIC = "user_emotion_topic"
OUT_QA_TOPIC = "system_text_response_topic"

# Below are the few shot examples.
FEW_SHOT_EXAMPLES = [
    {
        "question": "How do you find the source of the bleeding?",
        "answer": "Have the injured person lie down, which will make it easier to locate the exact source of the bleeding.",
    },
    {
        "question": "What should you do after identifying the source of bleeding?",
        "answer": "Apply direct pressure to the wound. If bleeding does not slow or stop after 15 minutes, consider using a tourniquet.",
    },
    {
        "question": "How should you position the tourniquet?",
        "answer": "Place the tourniquet on bare skin several inches above the injury, closer to the heart, and avoid placing it directly on a joint. Secure it with a common square knot.",
    },
    {
        "question": "What is a windlass, and how is it used in applying a tourniquet?",
        "answer": "A windlass is an object used to tighten the tourniquet. Place it on top of the square knot and tie the loose ends of the tourniquet around it with another square knot.",
    },
    {
        "question": "How do you tighten the tourniquet?",
        "answer": "Twist the windlass until the bleeding stops or is significantly reduced. Secure the windlass by tying one or both ends to the injured person's limb.",
    },
    {
        "question": "How long can a tourniquet be applied for?",
        "answer": "A tourniquet should not be applied for longer than two hours.",
    },
    {
        "question": "What should you do if the bleeding does not stop after applying a tourniquet?",
        "answer": "Try twisting the tourniquet more to see if it helps. If not, apply a second tourniquet immediately below the first one without removing the first one.",
    },
    {
        "question": "What are some common mistakes to avoid when applying a tourniquet?",
        "answer": "Common mistakes include waiting too long to apply a tourniquet, applying it too loosely, not applying a second tourniquet if needed, loosening a tourniquet, and leaving it on for too long.",
    },
    {
        "question": "Who should remove a tourniquet?",
        "answer": "A tourniquet should only be removed by a healthcare provider in the emergency department.",
    },
]
# Below is the expected formatting of each few shot example.
FEW_SHOT_EXAMPLE_TEMPLATE = """Question: {question}
Answer: {answer}
"""

# Below is the context of the prompt instructions. In this case, it isa specified recipe.
PROMPT_INSTRUCTIONS_CONTEXT = """I am wrapping a tourniquet for an injured comrade. Please help.
Here are the steps for wrapping a tourniquet.
'place tourniquet over affected extremity 2-3 inches above wound site (step 1)
'pull tourniquet tight (step 2)
'apply strap to strap body (step 3)
'turn windlass clock wise or counter clockwise until hemorrhage is controlled (step 4)
'lock windlass into the windlass keeper (step 5)'
'pull remaining strap over the windlass keeper (step 6)'
'secure strap and windlass keeper with keeper securing device (step 7)'
'mark time on securing device strap with permanent marker (step 8)'
"""
# Below is the complete set of prompt instructions.
PROMPT_INSTRUCTIONS = f"""{PROMPT_INSTRUCTIONS_CONTEXT}Here are commonly asked questions and answers about tourniquets. Answer only the last question.
"""
# Below is the suffix appended to the prompt to indicate to the LLM to answer.
INFERENCE_SAMPLE_SUFFIX = """Question: {question}
Answer:"""


class QuestionAnswerer(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (IN_EMOTION_TOPIC,),
                (OUT_QA_TOPIC,),
            ],
        )
        self._in_emotion_topic = param_values[IN_EMOTION_TOPIC]
        self._out_qa_topic = param_values[OUT_QA_TOPIC]

        self.question_queue = queue.Queue()
        self.handler_thread = threading.Thread(target=self.process_question_queue)
        self.handler_thread.start()

        self.is_openai_ready = True
        if not os.getenv("OPENAI_API_KEY"):
            self.log.info("OPENAI_API_KEY environment variable is unset!")
            self.is_openai_ready = False
        else:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not os.getenv("OPENAI_ORG_ID"):
            self.log.info("OPENAI_ORG_ID environment variable is unset!")
            self.is_openai_ready = False
        else:
            self.openai_org_id = os.getenv("OPENAI_ORG_ID")

        # Handle subscription/publication topics.
        self.subscription = self.create_subscription(
            InterpretedAudioUserEmotion,
            self._in_emotion_topic,
            self.question_answer_callback,
            1,
        )
        self._qa_publisher = self.create_publisher(
            SystemTextResponse, self._out_qa_topic, 1
        )
        self.chain = self._configure_langchain()

    def _configure_langchain(self):
        example_prompt = PromptTemplate(
            input_variables=["question", "answer"], template=FEW_SHOT_EXAMPLE_TEMPLATE
        )
        few_shot_prompt = FewShotPromptTemplate(
            examples=FEW_SHOT_EXAMPLES,
            example_prompt=example_prompt,
            prefix=PROMPT_INSTRUCTIONS,
            suffix=INFERENCE_SAMPLE_SUFFIX,
            input_variables=["question"],
            example_separator="\n",
        )
        openai_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key,
            temperature=0.0,
            max_tokens=64,
        )
        return LLMChain(llm=openai_llm, prompt=few_shot_prompt)

    def get_response(self, user_utterance: str, user_emotion: str):
        """
        Generate a  response to the utterance, enriched with the addition of
        the user's detected emotion. Inference calls can be added and revised
        here.
        """
        return_msg = ""
        try:
            if self.is_openai_ready:
                return_msg = colored(
                    self.chain.run(question=user_utterance) + "\n", "light_green"
                )
        except RuntimeError as err:
            self.log.info(err)
            colored_apology = colored(
                "I'm sorry. I don't know how to answer your statement.", "light_red"
            )
            colored_emotion = colored(user_emotion, "light_red")
            return_msg = (
                f"{colored_apology} I understand that you feel {colored_emotion}."
            )
        return return_msg

    def question_answer_callback(self, msg):
        """
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        """
        self.log.debug(f"Received message:\n\n{msg.utterance_text}")
        if not self._apply_filter(msg):
            return
        self.question_queue.put(msg)

    def process_question_queue(self):
        """
        Constant loop to process received questions.
        """
        while True:
            msg = self.question_queue.get()
            emotion = msg.user_emotion
            response = self.get_response(msg.utterance_text, emotion)
            self.publish_generated_response(msg.utterance_text, response)

    def publish_generated_response(self, utterance: str, response: str):
        msg = SystemTextResponse()
        msg.header.frame_id = "GPT Question Answering"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.utterance_text = utterance
        msg.response = response
        colored_utterance = colored(utterance, "light_blue")
        colored_response = colored(response, "light_green")
        self.log.info(
            f'Responding to utterance:\n>>> "{colored_utterance}"\n>>> with:\n'
            + f'>>> "{colored_response}"'
        )
        self._qa_publisher.publish(msg)

    def _apply_filter(self, msg):
        """
        Abstracts away any filtering to apply on received messages. Return
        none if the message should be filtered out. Else, return the incoming
        msg if it can be included.
        """
        return msg


def main():
    rclpy.init()
    question_answerer = QuestionAnswerer()
    rclpy.spin(question_answerer)
    question_answerer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
