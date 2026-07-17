from typing import Dict

try:
    from agents.base import BaseAgent
    from agents.gatewayAgent import GatewayAgent
    from agents.joinAgent import JoinAgent
    from agents.presentationAgent import PresentationAgent
    from agents.technicalNewsAgent import TechnicalNewsAgent
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.agents.gatewayAgent import GatewayAgent
    from backend.agents.joinAgent import JoinAgent
    from backend.agents.presentationAgent import PresentationAgent
    from backend.agents.technicalNewsAgent import TechnicalNewsAgent

AVAILABLE_AGENTS: Dict[str, BaseAgent] = {
    "gatewayAgent": GatewayAgent(),
    "presentationAgent": PresentationAgent(),
    "technicalNewsAgent": TechnicalNewsAgent(),
    "joinAgent": JoinAgent(),
}
