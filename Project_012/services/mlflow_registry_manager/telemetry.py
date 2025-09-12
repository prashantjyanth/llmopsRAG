from prometheus_fastapi_instrumentator import Instrumentator
def instrument(app):
  Instrumentator().instrument(app).expose(app, endpoint='/metrics'); return app
