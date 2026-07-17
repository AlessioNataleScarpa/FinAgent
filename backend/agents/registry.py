from typing import Dict

from agents.base import BaseAgent
from agents.gatewayAgent import GatewayAgent
from agents.joinAgent import JoinAgent
from agents.presentationAgent import PresentationAgent
from agents.technicalNewsAgent import TechnicalNewsAgent

AVAILABLE_AGENTS: Dict[str, BaseAgent] = {
    "gatewayAgent": GatewayAgent(),
    "presentationAgent": PresentationAgent(),
    "technicalNewsAgent": TechnicalNewsAgent(),
    "joinAgent": JoinAgent(),
}
