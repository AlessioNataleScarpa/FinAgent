from typing import Dict

from agents.base import BaseAgent
from agents.gatewayAgent import GatewayAgent

AVAILABLE_AGENTS: Dict[str, BaseAgent] = {
    "gatewayAgent": GatewayAgent(),
}
