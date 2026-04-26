import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Mocking FastMCP and other dependencies before importing server
mock_mcp = MagicMock()
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = MagicMock()
sys.modules["mcp.server.fastmcp"].FastMCP = MagicMock(return_value=mock_mcp)


# Mock decorative tools
def mock_decorator(*args, **kwargs):
    def wrapper(func):
        return func

    return wrapper


mock_mcp.tool = mock_decorator
mock_mcp.resource = mock_decorator

sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.storage"] = MagicMock()
sys.modules["google.cloud.logging"] = MagicMock()
sys.modules["google.cloud.secretmanager"] = MagicMock()

# Now import the functions to test
from server import (  # noqa: E402
    MODEL_NAME,
    get_vllm_deployment_config,
    get_vllm_model_stats,
    query_vllm_with_metrics,
    save_hf_token,
    verify_model_health,
)


class TestDevOpsAgent(unittest.IsolatedAsyncioTestCase):
    def test_model_name_default(self):
        """Verify the default model is Gemma 4."""
        self.assertEqual(MODEL_NAME, "google/gemma-4-31B-it")

    @patch("server.get_secret", new_callable=AsyncMock)
    async def test_get_vllm_deployment_config(self, mock_get_secret):
        """Test TPU deployment config generation."""
        mock_get_secret.return_value = "dummy-hf-token"
        config = await get_vllm_deployment_config(service_name="test-vllm", model_name="google/gemma-4-31B-it")
        self.assertIn("gcloud alpha compute tpus tpu-vm create test-vllm", config)
        self.assertIn("--accelerator-type=v6e-8", config)
        self.assertIn("--version=v2-alpha-tpuv6e", config)

        self.assertIn("vllm/vllm-tpu:nightly", config)
        self.assertIn("google/gemma-4-31B-it", config)

    @patch("server.get_vllm_client", new_callable=AsyncMock)
    @patch("server.discover_vllm_url", new_callable=AsyncMock)
    async def test_verify_model_health_success(self, mock_discover_url, mock_get_client):
        """Test successful model health check."""
        mock_discover_url.return_value = "http://test-url:8000"
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "READY"

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await verify_model_health()
        self.assertIn("PASSED", result)
        self.assertIn("READY", result)

    @patch("server.get_vllm_client", new_callable=AsyncMock)
    async def test_query_vllm_with_metrics_success(self, mock_get_client):
        """Test query with performance metrics."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_stream = AsyncMock()

        # Create chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        mock_stream.__aiter__.return_value = [chunk1, chunk2]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        result = await query_vllm_with_metrics("Hi")
        self.assertIn("Hello world", result)
        self.assertIn("Performance Metrics", result)
        self.assertIn("TTFT", result)

    @patch("server.discover_vllm_url", new_callable=AsyncMock)
    @patch("httpx.AsyncClient", autospec=True)
    async def test_get_vllm_model_stats_success(self, mock_async_client, mock_discover_url):
        """Test retrieving model stats."""
        mock_discover_url.return_value = "http://test-url:8000"

        # Mock httpx.AsyncClient and its get method
        mock_client_instance = mock_async_client.return_value.__aenter__.return_value

        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        mock_models_response = MagicMock()
        mock_models_response.json.return_value = {"data": [{"id": "test-model", "max_model_len": 4096}]}
        mock_models_response.status_code = 200

        mock_client_instance.get.side_effect = [mock_health_response, mock_models_response]

        result = await get_vllm_model_stats()
        self.assertIn("### 📈 Model Statistics", result)
        self.assertIn("test-model", result)
        self.assertIn("4096", result)

    @patch("server.secretmanager.SecretManagerServiceClient")
    async def test_save_hf_token(self, mock_secret_client):
        """Test saving HF token to Secret Manager."""
        mock_instance = mock_secret_client.return_value
        mock_instance.add_secret_version.return_value.name = "projects/test-project/secrets/hf-token/versions/1"

        # Test successful save (secret exists)
        result = await save_hf_token("test-token")
        self.assertIn("✅ Token saved.", result)

        mock_instance.add_secret_version.assert_called()


if __name__ == "__main__":
    unittest.main()
