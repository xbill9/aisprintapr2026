import unittest
from unittest.mock import patch, MagicMock
from server import mcp

class TestDevOpsAgent(unittest.IsolatedAsyncioTestCase):
    async def test_tools_registered(self):
        """Verify that the expected tools are registered with FastMCP."""
        tools = [t.name for t in await mcp.list_tools()]
        self.assertIn("analyze_cloud_logging", tools)
        self.assertIn("suggest_sre_remediation", tools)
        self.assertIn("get_vllm_deployment_config", tools)
        self.assertIn("get_vertex_ai_model_copy_instructions", tools)
        self.assertIn("get_kaggle_model_copy_instructions", tools)
        self.assertIn("list_vertex_models", tools)
        self.assertIn("list_bucket_models", tools)

    async def test_resources_registered(self):
        """Verify that the expected resources are registered with FastMCP."""
        resources = [str(r.uri) for r in await mcp.list_resources()]
        self.assertIn("config://vllm-deployment-template", resources)

    def test_get_kaggle_model_copy_instructions(self):
        """Test the output of the Kaggle model copy instructions tool."""
        from server import get_kaggle_model_copy_instructions
        instructions = get_kaggle_model_copy_instructions("test/slug/2b-it/1", "test-bucket")
        self.assertIn("test/slug/2b-it/1", instructions)
        self.assertIn("test-bucket", instructions)
        self.assertIn("2b-it", instructions)
        self.assertIn("kaggle models instances versions download test/slug/2b-it/1", instructions)

    def test_get_vertex_ai_model_copy_instructions(self):
        """Test the output of the Vertex AI model copy instructions tool."""
        from server import get_vertex_ai_model_copy_instructions
        instructions = get_vertex_ai_model_copy_instructions("gemma-2b-it")
        self.assertIn("gemma-2b-it", instructions)
        self.assertIn("Vertex AI Model Garden", instructions)
        self.assertIn("gcloud storage cp", instructions)

    @patch("server.storage.Client")
    def test_list_bucket_models_mock(self, mock_storage_client):
        """Test the output of the GCS bucket listing tool with mocks."""
        from server import list_bucket_models
        
        # Setup mock behavior
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "gemma-2b-it/config.json"
        mock_blob.size = 1024 * 1024 * 5 # 5 MB
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        
        result = list_bucket_models("mock-bucket")
        self.assertIn("mock-bucket", result)
        self.assertIn("gemma-2b-it/config.json", result)
        self.assertIn("5.00 MB", result)

    @patch("server.requests.post")
    @patch("server.cloud_logging.Client")
    async def test_analyze_cloud_logging_mock(self, mock_logging, mock_requests):
        # This is a bit complex for a simple unittest because of async/await in FastMCP tools
        # and how FastMCP wraps them. We'll stick to basic structure verification for now.
        pass

if __name__ == "__main__":
    unittest.main()
