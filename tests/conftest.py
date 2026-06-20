import pytest

# Run async tests only on the asyncio backend (trio is not installed)
@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param
