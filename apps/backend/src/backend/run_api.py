import uvicorn
from dotenv import load_dotenv
from shared.core import settings

load_dotenv()


def main() -> None:
    uvicorn.run(
        "backend.api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_dev(),
    )


if __name__ == "__main__":
    main()
