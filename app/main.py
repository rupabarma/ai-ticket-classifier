"""
TicketFlow - FastAPI application.

Start from the project root:

    uvicorn app.main:app --reload

All pages are normal server-rendered HTML. Forms submit with a regular POST
and the page reloads with the result (no JavaScript).
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services import TicketService

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: creates dataset/model if missing, then loads them.
    app.state.service = TicketService()
    yield


app = FastAPI(title="TicketFlow", description="AI-powered support ticket classification", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def render(request: Request, template: str, context: dict, status_code: int = 200):
    return templates.TemplateResponse(request=request, name=template, context=context, status_code=status_code)


@app.get("/")
async def dashboard(request: Request):
    return render(request, "dashboard.html", request.app.state.service.dashboard_context())


@app.get("/classify")
async def classify_form(request: Request):
    return render(request, "classify.html", request.app.state.service.classify_page_context())


@app.post("/classify")
async def classify_ticket(request: Request, subject: str = Form(""), body: str = Form("")):
    context = request.app.state.service.classify_submission(subject, body)
    status = 422 if context["errors"] else 200
    return render(request, "classify.html", context, status_code=status)


@app.get("/metrics")
async def metrics(request: Request):
    return render(request, "metrics.html", request.app.state.service.metrics_context())


@app.get("/about")
async def about(request: Request):
    return render(request, "about.html", request.app.state.service.about_context())
