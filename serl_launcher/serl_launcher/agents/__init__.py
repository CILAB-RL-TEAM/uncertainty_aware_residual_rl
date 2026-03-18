from .continuous.bc import BCAgent
from .continuous.assymetric_sac import SACAgent

agents = {
    "bc": BCAgent,
    "sac": SACAgent,
}
