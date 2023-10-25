
import langchain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import openai
from operator import itemgetter
import os
import queue
import rclpy
from rclpy.node import Node
import requests
from termcolor import colored
import threading

from angel_msgs.msg import ActivityDetection, InterpretedAudioUserEmotion, ObjectDetection2dSet, SystemTextResponse
from angel_utils import declare_and_get_parameters

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

IN_EMOTION_TOPIC = "user_emotion_topic"
IN_OBJECT_DETECTION_TOPIC = "object_detections_topic"
IN_ACT_CLFN_TOPIC = "action_classifications_topic"
OUT_QA_TOPIC = "system_text_response_topic"
IN_OBJECT_DETECTION_THRESHOLD = "object_detections_threshold"
IN_ACT_CLFN_THRESHOLD = "action_classification_threshold"

# Below is the complete set of prompt instructions.
PROMPT_INSTRUCTIONS = """
You will be given a User Scenario. All the objects in front of and observable to the user are included.
Your task is to use the Action Steps to answer the user's Question.

Action Steps:
Step 1. Place tourniquet over affected extremity 2-3 inches above wound site.
Step 2. Pull tourniquet tight.
Step 3. Apply strap to strap body.
Step 4. Turn the windlass clock wise or counter clockwise until hemorrhage is controlled.
Step 5. Lock the windlass into the windlass keeper.
Step 6. Pull remaining strap over the windlass keeper.
Step 7. Secure strap and windlass keeper with keeper securing device.
Step 8. Mark the time on securing device strap with permanent marker. You are complete.

User Scenario:
The User is doing {action}. The User can see {observables}.

User Question: {question}
Answer:"""
# Below is the suffix appended to the prompt to indicate to the LLM to answer.
INFERENCE_SAMPLE_SUFFIX = """Question: {question}
Answer:"""

langchain.debug = True

class VisualQuestionAnswerer(Node):

    class TimestampedEntity:

        def __init__(self, time, entity: str):
            self.time = time
            self.entity = entity

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (IN_EMOTION_TOPIC,),
                (IN_OBJECT_DETECTION_TOPIC, ""),
                (IN_ACT_CLFN_TOPIC, ""),
                (IN_OBJECT_DETECTION_THRESHOLD, 0.8),
                (IN_ACT_CLFN_THRESHOLD, 0.8),
                (OUT_QA_TOPIC,),
            ],
        )
        self._in_emotion_topic = param_values[IN_EMOTION_TOPIC]
        self._in_objects_topic = param_values[IN_OBJECT_DETECTION_TOPIC]
        self._in_actions_topic = param_values[IN_ACT_CLFN_TOPIC]
        self._out_qa_topic = param_values[OUT_QA_TOPIC]
        self._object_dtctn_threshold = param_values[IN_OBJECT_DETECTION_THRESHOLD]
        self._action_clfn_threshold = param_values[IN_ACT_CLFN_THRESHOLD]

        self.question_queue = queue.Queue()
        self.action_classification_queue = queue.Queue()
        self.detected_objects_queue = queue.Queue()

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

        # Configure the (necessary) emotional detection enriched utterance subscription.
        self.emotion_subscription = self.create_subscription(
            InterpretedAudioUserEmotion,
            self._in_emotion_topic,
            self.question_answer_callback,
            1,
        )
        # Configure the optional object detection subscription.
        self.objects_subscription = None
        if self._in_emotion_topic:
            self.objects_subscription = self.create_subscription(
                ObjectDetection2dSet,
                self._in_objects_topic,
                self._add_detected_objects,
                1,
            )
        # Configure the optional action classification subscription.           
        self.action_subscription = None
        if self.action_subscription:
            self.action_subscription = self.create_subscription(
                ActivityDetection,
                self._in_actions_topic,
                self._add_action_classification,
                1,
            )
        # Configure the sole QA output of this node.
        self._qa_publisher = self.create_publisher(
            SystemTextResponse, self._out_qa_topic, 1
        )

        self.chain = self._configure_langchain()

    def _configure_langchain(self):
        openai_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key,
            temperature=0.0,
            max_tokens=64,
        )
        zero_shot_prompt = langchain.PromptTemplate(
            input_variables=["action", "observables", "question"],
            template=PROMPT_INSTRUCTIONS,
        )
        return LLMChain(llm=openai_llm, prompt=zero_shot_prompt)

    def _get_sec(self, msg) -> int:
        return msg.header.stamp.sec

    def _add_action_classification(self, msg: ActivityDetection) -> str:
        '''
        Stores the action label with the highest confidence in
        self.action_classification_queue.
        '''
        print(msg)
        action_classification = max(zip(msg.label_vec, msg.conf_vec), key = itemgetter(1))[0]
        te = VisualQuestionAnswerer.TimestampedEntity(self._get_sec(msg), action_classification)
        self.action_classification_queue.put(te)

    def _add_detected_objects(self, msg: ObjectDetection2dSet) -> str:
        '''
        Stores all items with a confidence score above IN_OBJECT_DETECTION_THRESHOLD.
        '''
        detected_objects = set()
        for obj, score in zip(msg.label_vec, msg.label_confidences):
            if score < self._object_dtctn_threshold:
                # Optional threshold filtering
                continue
            detected_objects.add(obj)
        if detected_objects:
            te = VisualQuestionAnswerer.TimestampedEntity(self._get_sec(msg), detected_objects)
            self.detected_objects_queue.put(te)

    def _get_action_before(self, curr_time: int) -> str:
        '''
        Returns the latest action classification in self.action_classification_queue
        that does not occur before a provided time.
        '''
        latest_action = "nothing"
        while not self.action_classification_queue.empty():
            next = self.action_classification_queue.queue[0]
            if next.time < curr_time:
                latest_action = next.entity
                self.action_classification_queue.get()
            else:
                break
        return latest_action

    def _get_observables_before(self, curr_time: int) -> str:
        '''
        Returns a comma-delimited list of observed objects per all
        entities in self.detected_objects_queue that occurred before a provided time.
        '''
        observables = set()
        while not self.detected_objects_queue.empty():
            next = self.detected_objects_queue.queue[0]
            if next.time < curr_time:
                observables.update(next.entity)
                self.detected_objects_queue.get()
            else:
                break
        if not observables:
            return "nothing"
        return ", ".join(list(observables))                

    def get_response(self, msg: InterpretedAudioUserEmotion):
        """
        Generate a  response to the utterance, enriched with the addition of
        the user's detected emotion. Inference calls can be added and revised
        here.
        """
        return_msg = ""
        try:
            # Apply detected actions.
            action = self._get_action_before(self._get_sec(msg))
            # print(f"Latest action: {action}")
            # Apply detected objects.
            observables = self._get_observables_before(self._get_sec(msg))
            # print(f"Observed objects: {observables}")
            response = self.chain.run(
                action=action, observables=observables, question=msg.utterance_text
            )
            return_msg = colored(f"{response}\n", "light_green")
        except RuntimeError as err:
            self.log.info(err)
            colored_apology = colored("I'm sorry. I don't know how to answer your statement.",
                                      "light_red")
            colored_emotion = colored(msg.user_emotion, "light_red")
            return_msg = f"{colored_apology} I understand that you feel {colored_emotion}."
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
            response = self.get_response(msg)
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
    question_answerer = VisualQuestionAnswerer()
    rclpy.spin(question_answerer)
    question_answerer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
