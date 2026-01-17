from api.config import get_settings
from api.services.research_service import ResearchService
from api.services.task_manager import TaskManager


task_manager = TaskManager()
research_service = ResearchService(get_settings(), task_manager)
